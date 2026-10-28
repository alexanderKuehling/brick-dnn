from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('brick_dnn_core', '0002_training'),
    ]

    operations = [
        migrations.AddField(
            model_name='trainingrun',
            name='tuning',
            field=models.JSONField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='trainingrun',
            name='trials_done',
            field=models.PositiveIntegerField(default=0),
        ),
        migrations.AddField(
            model_name='trainingrun',
            name='best_trial',
            field=models.JSONField(blank=True, null=True),
        ),
    ]
