from django.contrib import admin
from .models import Tablet

@admin.register(Tablet)
class TabletAdmin(admin.ModelAdmin):
    list_display = ('user', 'name', 'count', 'dosage', 'start_date', 'finish_date')