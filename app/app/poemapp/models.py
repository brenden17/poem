from __future__ import unicode_literals
from django.utils.encoding import python_2_unicode_compatible

from django.db import models
from django.contrib.auth.models import User

class PublishedManager(models.Manager):
    def get_queryset(self):
        return super(PublishedManager, self).get_queryset().\
                                            filter(status='published')

@python_2_unicode_compatible
class Poem(models.Model):
    STATUS_CHOICE = (
        ('conform', 'Conform'),
        ('progress', 'Progress'),
    )
    DECISION_CHOICE = (
        ('yes', 'Yes'),
        ('no', 'no'),
    )
    title = models.CharField(max_length=100)
    author = models.CharField(max_length=100)
    content = models.TextField()
    slug = models.CharField(max_length=10)
    yes_count = models.IntegerField(default=0)
    no_count = models.IntegerField(default=0)
    prediction = models.CharField(max_length=10, 
                                choices=DECISION_CHOICE,
                                default='no')
    evaluation = models.CharField(max_length=10, 
                                choices=DECISION_CHOICE,
                                default='no')
    newspaper = models.CharField(max_length=100)
    page = models.CharField(max_length=10)
    published = models.DateField()
    prediction_score = models.FloatField(default=0.0)
    status = models.CharField(max_length=10, 
                                choices=STATUS_CHOICE,
                                default='progress')
    user = models.ForeignKey(User, 
                                related_name='poem',
                                blank=True,
                                null=True)
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)

    objects = models.Manager()
    publish = PublishedManager()

    class Meta:
        ordering = ('-published',)    

    def __str__(self):
        return '{}, {}, {}/{}'.format(self.title,
                                            self.author,
                                            self.newspaper,
                                            self.published)

    def get_absolute_url(self):
        return reverse('poem:poem_detail', args=(self.slug,))
