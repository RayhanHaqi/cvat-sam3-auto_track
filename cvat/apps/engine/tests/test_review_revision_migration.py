"""An old Completed job must not acquire a review binding on migration."""

from django.db import connection
from django.db.migrations.executor import MigrationExecutor
from django.test import TransactionTestCase

from cvat.apps.engine.models import Job


class JobReviewMigrationTest(TransactionTestCase):
    def test_existing_completed_job_stays_unbound(self):
        before = ("engine", "0098_data_local_storage_backing_cs")
        after = ("engine", "0099_job_annotation_review_revision")
        executor = MigrationExecutor(connection)
        executor.migrate([before])
        try:
            apps = executor.loader.project_state([before]).apps
            task = apps.get_model("engine", "Task").objects.create(
                name="legacy completed job", mode="annotation"
            )
            segment = apps.get_model("engine", "Segment").objects.create(
                task_id=task.pk, start_frame=0, stop_frame=0
            )
            job_id = apps.get_model("engine", "Job").objects.create(
                segment_id=segment.pk, state="completed"
            ).pk
        finally:
            MigrationExecutor(connection).migrate([after])

        job = Job.objects.get(pk=job_id)
        self.assertEqual(job.state, "completed")
        self.assertEqual(job.annotation_revision, 0)
        self.assertIsNone(job.completed_annotation_revision)
