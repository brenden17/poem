# -*- coding: utf-8 -*-
# Generated by Django 1.9.5 on 2016-04-14 10:15
from __future__ import unicode_literals

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='Poem',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('title', models.CharField(max_length=100)),
                ('author', models.CharField(max_length=100)),
                ('content', models.TextField()),
                ('slug', models.CharField(max_length=10)),
                ('yes_count', models.IntegerField(default=0)),
                ('no_count', models.IntegerField(default=0)),
                ('prediction', models.CharField(choices=[('conform', 'Conform'), ('progress', 'Progress')], default='no', max_length=10)),
                ('evaluation', models.CharField(choices=[('conform', 'Conform'), ('progress', 'Progress')], default='no', max_length=10)),
                ('newspaper', models.CharField(max_length=100)),
                ('page', models.CharField(max_length=10)),
                ('published', models.DateField()),
                ('prediction_score', models.FloatField(default=0.0)),
                ('status', models.CharField(choices=[('conform', 'Conform'), ('progress', 'Progress')], default='progress', max_length=10)),
                ('created', models.DateTimeField(auto_now_add=True)),
                ('updated', models.DateTimeField(auto_now=True)),
                ('user', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='poem', to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'ordering': ('-published',),
            },
        ),
    ]
