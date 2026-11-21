"""Initial HopeXAI persistent state schema.

This baseline creates all tables from the reviewed SQLAlchemy metadata. New
releases should add explicit Alembic operations in subsequent revisions.
"""

from alembic import op
from app.models import Base
import app.models  # noqa: F401

revision = "0001_initial_schema"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    Base.metadata.create_all(bind=op.get_bind(), checkfirst=True)


def downgrade() -> None:
    Base.metadata.drop_all(bind=op.get_bind(), checkfirst=True)
