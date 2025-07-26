from django.db import models
from django.contrib.auth.models import User

class Tablet(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='tablets')
    name = models.CharField(max_length=100)
    count = models.IntegerField()
    dosage = models.CharField(max_length=10)  # 'full' or 'half'
    start_date = models.DateField()
    finish_date = models.DateField()

    def __str__(self):
        return f"{self.name} ({self.user.username})"



