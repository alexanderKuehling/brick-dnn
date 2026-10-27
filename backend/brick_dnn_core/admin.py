from django.contrib import admin

from .models import Project


@admin.register(Project)
class ProjectAdmin(admin.ModelAdmin):
    list_display = ("name", "owner", "updated_at")
    list_filter = ("owner",)
    search_fields = ("name", "description", "owner__username")
