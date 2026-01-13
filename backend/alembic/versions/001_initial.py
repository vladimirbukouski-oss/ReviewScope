"""Initial migration - create all tables

Revision ID: 001_initial
Revises:
Create Date: 2025-01-13

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '001_initial'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create analyses table
    op.create_table(
        'analyses',
        sa.Column('id', sa.String(16), primary_key=True),
        sa.Column('url', sa.String(512), nullable=False),
        sa.Column('cache_key', sa.String(32), nullable=False),
        sa.Column('service', sa.String(20), nullable=False),
        sa.Column('status', sa.String(20), default='pending'),
        sa.Column('progress', sa.Integer(), default=0),
        sa.Column('message', sa.String(255), default=''),
        sa.Column('error', sa.Text(), nullable=True),
        sa.Column('product_name', sa.String(512), nullable=True),
        sa.Column('product_image', sa.String(512), nullable=True),
        sa.Column('truth_stars', sa.Float(), nullable=True),
        sa.Column('avg_orig_stars', sa.Float(), nullable=True),
        sa.Column('suspicious_share', sa.Float(), nullable=True),
        sa.Column('total_reviews', sa.Integer(), nullable=True),
        sa.Column('kept_reviews', sa.Integer(), nullable=True),
        sa.Column('summary_data', sa.JSON(), nullable=True),
        sa.Column('stage4_summary', sa.JSON(), nullable=True),
        sa.Column('sentiment_mix', sa.JSON(), nullable=True),
        sa.Column('bundle_path', sa.String(512), nullable=True),
        sa.Column('rag_dir', sa.String(512), nullable=True),
        sa.Column('created_at', sa.DateTime(), default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), default=sa.func.now(), onupdate=sa.func.now()),
    )
    op.create_index('ix_analyses_url', 'analyses', ['url'])
    op.create_index('ix_analyses_cache_key', 'analyses', ['cache_key'])
    op.create_index('ix_analyses_created_at', 'analyses', ['created_at'])
    op.create_index('ix_analyses_cache_status', 'analyses', ['cache_key', 'status'])

    # Create users table
    op.create_table(
        'users',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('email', sa.String(255), unique=True, nullable=False),
        sa.Column('password_hash', sa.String(255), nullable=False),
        sa.Column('name', sa.String(100), nullable=True),
        sa.Column('is_active', sa.Boolean(), default=True),
        sa.Column('is_verified', sa.Boolean(), default=False),
        sa.Column('notify_email', sa.Boolean(), default=True),
        sa.Column('notify_telegram', sa.Boolean(), default=False),
        sa.Column('telegram_chat_id', sa.String(50), nullable=True),
        sa.Column('created_at', sa.DateTime(), default=sa.func.now()),
        sa.Column('last_login', sa.DateTime(), nullable=True),
    )
    op.create_index('ix_users_email', 'users', ['email'])

    # Create user_analyses table
    op.create_table(
        'user_analyses',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('user_id', sa.Integer(), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        sa.Column('analysis_id', sa.String(16), sa.ForeignKey('analyses.id', ondelete='CASCADE'), nullable=False),
        sa.Column('is_favorite', sa.Boolean(), default=False),
        sa.Column('viewed_at', sa.DateTime(), default=sa.func.now()),
    )
    op.create_index('ix_user_analyses_user_id', 'user_analyses', ['user_id'])
    op.create_index('ix_user_analyses_analysis_id', 'user_analyses', ['analysis_id'])
    op.create_index('ix_user_analyses_user_favorite', 'user_analyses', ['user_id', 'is_favorite'])

    # Create tracked_products table
    op.create_table(
        'tracked_products',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('user_id', sa.Integer(), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        sa.Column('url', sa.String(512), nullable=False),
        sa.Column('service', sa.String(20), nullable=False),
        sa.Column('product_name', sa.String(512), nullable=True),
        sa.Column('is_active', sa.Boolean(), default=True),
        sa.Column('check_interval_hours', sa.Integer(), default=24),
        sa.Column('notify_price_drop_percent', sa.Float(), default=10.0),
        sa.Column('notify_rating_change', sa.Float(), default=0.3),
        sa.Column('notify_new_reviews', sa.Integer(), default=10),
        sa.Column('last_price', sa.Float(), nullable=True),
        sa.Column('last_rating', sa.Float(), nullable=True),
        sa.Column('last_review_count', sa.Integer(), nullable=True),
        sa.Column('last_check_at', sa.DateTime(), nullable=True),
        sa.Column('latest_analysis_id', sa.String(16), sa.ForeignKey('analyses.id', ondelete='SET NULL'), nullable=True),
        sa.Column('created_at', sa.DateTime(), default=sa.func.now()),
    )
    op.create_index('ix_tracked_products_user_id', 'tracked_products', ['user_id'])

    # Create analysis_views table
    op.create_table(
        'analysis_views',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('analysis_id', sa.String(16), sa.ForeignKey('analyses.id', ondelete='CASCADE'), nullable=False),
        sa.Column('ip_hash', sa.String(64), nullable=True),
        sa.Column('user_agent', sa.String(255), nullable=True),
        sa.Column('referer', sa.String(512), nullable=True),
        sa.Column('viewed_at', sa.DateTime(), default=sa.func.now()),
    )
    op.create_index('ix_analysis_views_analysis_id', 'analysis_views', ['analysis_id'])
    op.create_index('ix_analysis_views_viewed_at', 'analysis_views', ['viewed_at'])
    op.create_index('ix_analysis_views_date', 'analysis_views', ['analysis_id', 'viewed_at'])


def downgrade() -> None:
    op.drop_table('analysis_views')
    op.drop_table('tracked_products')
    op.drop_table('user_analyses')
    op.drop_table('users')
    op.drop_table('analyses')
