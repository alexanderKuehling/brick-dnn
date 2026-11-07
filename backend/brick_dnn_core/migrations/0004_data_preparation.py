import brick_dnn_core.models
import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('brick_dnn_core', '0003_training_split_and_search'),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='DataPrep',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=120)),
                ('steps', models.JSONField(default=list)),
                ('settings', models.JSONField(default=brick_dnn_core.models.default_prep_settings)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('owner', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='data_preps', to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'ordering': ['-updated_at', '-id'],
            },
        ),
        migrations.AddField(
            model_name='dataset',
            name='prepared_from',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.SET_NULL, related_name='datasets', to='brick_dnn_core.dataprep'),
        ),
        migrations.CreateModel(
            name='DataSource',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=120)),
                ('filename', models.CharField(blank=True, max_length=255)),
                ('size_bytes', models.BigIntegerField(default=0)),
                ('n_rows', models.BigIntegerField(default=0)),
                ('n_columns', models.PositiveIntegerField(default=0)),
                ('columns', models.JSONField(default=list)),
                ('import_options', models.JSONField(default=dict)),
                ('report', models.JSONField(default=dict)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('owner', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='data_sources', to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'ordering': ['-created_at', '-id'],
            },
        ),
        migrations.AddField(
            model_name='dataprep',
            name='source',
            field=models.ForeignKey(on_delete=django.db.models.deletion.RESTRICT, related_name='preps', to='brick_dnn_core.datasource'),
        ),
    ]
