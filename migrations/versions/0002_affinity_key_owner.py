"""Associate conversation affinity with its HopeXAI API key for revocation cleanup."""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect

revision = "0002_affinity_key_owner"
down_revision = "0001_initial_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    if "conversation_affinity" not in inspector.get_table_names():
        return
    columns = {column["name"] for column in inspector.get_columns("conversation_affinity")}
    if "api_key_id" not in columns:
        op.add_column("conversation_affinity", sa.Column("api_key_id", sa.String(length=36), nullable=True))
    indexes = {index["name"] for index in inspect(bind).get_indexes("conversation_affinity")}
    if "ix_affinity_key_owner" not in indexes:
        op.create_index("ix_affinity_key_owner", "conversation_affinity", ["api_key_id"], unique=False)
    op.execute("DELETE FROM conversation_affinity WHERE api_key_id IS NULL")


def downgrade() -> None:
    bind = op.get_bind()
    inspector = inspect(bind)
    if "conversation_affinity" not in inspector.get_table_names():
        return
    indexes = {index["name"] for index in inspector.get_indexes("conversation_affinity")}
    if "ix_affinity_key_owner" in indexes:
        op.drop_index("ix_affinity_key_owner", table_name="conversation_affinity")
    columns = {column["name"] for column in inspect(bind).get_columns("conversation_affinity")}
    if "api_key_id" in columns:
        if bind.dialect.name == "sqlite":
            with op.batch_alter_table("conversation_affinity") as batch:
                batch.drop_column("api_key_id")
        else:
            op.drop_column("conversation_affinity", "api_key_id")
