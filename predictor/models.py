from django.db import models


class Person(models.Model):
    name = models.CharField(max_length=30)
    age = models.IntegerField()
    salary = models.IntegerField()
    purchase = models.BooleanField(default=False)

    def __str__(self):
        return self.name