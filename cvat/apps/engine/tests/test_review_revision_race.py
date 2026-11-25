"""The Completed binding and annotation revision share one PostgreSQL lock."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event
from unittest import skipUnless

from django.db import close_old_connections, connection, transaction
from django.test import TransactionTestCase

from cvat.apps.dataset_manager.task import JobAnnotation
from cvat.apps.engine.models import Job, Segment, StateChoice, Task
from cvat.apps.engine.serializers import JobWriteSerializer


@skipUnless(connection.vendor == "postgresql", "requires PostgreSQL row locks")
class JobReviewConcurrencyTest(TransactionTestCase):
    def test_completion_and_annotation_write_are_serialized(self):
        task = Task.objects.create(name="isolated review race", mode="annotation")
        segment = Segment.objects.create(task=task, start_frame=0, stop_frame=0)

        for first in ("complete", "edit"):
            job = Job.objects.create(segment=segment)
            locked, attempted, release = Event(), Event(), Event()

            def write(kind):
                if kind == "complete":
                    serializer = JobWriteSerializer(
                        Job.objects.get(pk=job.pk),
                        data={"state": StateChoice.COMPLETED}, partial=True,
                    )
                    serializer.is_valid(raise_exception=True)
                    serializer.save()
                else:
                    JobAnnotation(job.pk, db_job=Job.objects.get(pk=job.pk))._set_updated_date()

            def first_writer():
                close_old_connections()
                try:
                    with transaction.atomic():
                        write(first)
                        locked.set()
                        if not release.wait(10):
                            raise TimeoutError("second writer did not reach the lock")
                finally:
                    close_old_connections()

            def second_writer():
                close_old_connections()
                try:
                    if not locked.wait(10):
                        raise TimeoutError("first writer never acquired the lock")

                    def track_lock(execute, sql, params, many, context):
                        if "FOR UPDATE" in sql and "engine_job" in sql:
                            attempted.set()
                        return execute(sql, params, many, context)

                    with connection.execute_wrapper(track_lock):
                        write("edit" if first == "complete" else "complete")
                finally:
                    close_old_connections()

            with ThreadPoolExecutor(max_workers=2) as pool:
                held = pool.submit(first_writer)
                waiting = pool.submit(second_writer)
                try:
                    self.assertTrue(attempted.wait(10), "second writer did not attempt row lock")
                    self.assertFalse(waiting.done(), "job row lock was not held")
                finally:
                    release.set()
                held.result(timeout=10)
                waiting.result(timeout=10)

            job.refresh_from_db()
            self.assertEqual(job.annotation_revision, 1)
            self.assertEqual(
                (job.state, job.completed_annotation_revision),
                (StateChoice.IN_PROGRESS, None) if first == "complete"
                else (StateChoice.COMPLETED, 1),
            )
