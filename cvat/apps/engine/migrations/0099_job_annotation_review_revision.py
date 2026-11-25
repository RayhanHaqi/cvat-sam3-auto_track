from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [("engine", "0098_data_local_storage_backing_cs")]

    operations = [
        migrations.AddField(
            model_name="job",
            name="annotation_revision",
            field=models.PositiveBigIntegerField(default=0),
        ),
        migrations.AddField(
            model_name="job",
            name="completed_annotation_revision",
            field=models.PositiveBigIntegerField(default=None, null=True),
        ),
    ]
