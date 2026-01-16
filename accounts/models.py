from django.contrib.auth.models import AbstractUser
from django.db import models


class CustomUser(AbstractUser):
    """
    Custom user model for future extensibility.

    Extending AbstractUser allows adding business-specific fields later
    (e.g., phone_number, subscription_tier) without database migration issues.
    """
    pass
