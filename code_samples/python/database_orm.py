"""
Database and ORM Examples in Python
Demonstrates SQLAlchemy, database design, migrations, and advanced queries.
"""

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Boolean, DateTime, 
    ForeignKey, Table, text, func, and_, or_, desc
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, joinedload
from sqlalchemy.pool import StaticPool
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import contextlib
from dataclasses import dataclass
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database setup
Base = declarative_base()

# Association table for many-to-many relationship
user_roles = Table(
    'user_roles',
    Base.metadata,
    Column('user_id', Integer, ForeignKey('users.id'), primary_key=True),
    Column('role_id', Integer, ForeignKey('roles.id'), primary_key=True)
)


# Model Definitions
class User(Base):
    """User model with relationships."""
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False, index=True)
    full_name = Column(String(100))
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    profile = relationship("UserProfile", back_populates="user", uselist=False, cascade="all, delete-orphan")
    posts = relationship("Post", back_populates="author", cascade="all, delete-orphan")
    comments = relationship("Comment", back_populates="author", cascade="all, delete-orphan")
    roles = relationship("Role", secondary=user_roles, back_populates="users")
    
    def __repr__(self):
        return f"<User(id={self.id}, username='{self.username}', email='{self.email}')>"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert user to dictionary."""
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'full_name': self.full_name,
            'is_active': self.is_active,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


class UserProfile(Base):
    """User profile model (one-to-one with User)."""
    __tablename__ = 'user_profiles'
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id'), unique=True, nullable=False)
    bio = Column(String(500))
    avatar_url = Column(String(200))
    birth_date = Column(DateTime)
    location = Column(String(100))
    website = Column(String(200))
    
    # Relationship
    user = relationship("User", back_populates="profile")
    
    def __repr__(self):
        return f"<UserProfile(id={self.id}, user_id={self.user_id})>"


class Role(Base):
    """Role model for user permissions."""
    __tablename__ = 'roles'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(50), unique=True, nullable=False)
    description = Column(String(200))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    users = relationship("User", secondary=user_roles, back_populates="roles")
    
    def __repr__(self):
        return f"<Role(id={self.id}, name='{self.name}')>"


class Category(Base):
    """Category model for posts."""
    __tablename__ = 'categories'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(50), unique=True, nullable=False)
    description = Column(String(200))
    parent_id = Column(Integer, ForeignKey('categories.id'))
    
    # Self-referential relationship for hierarchical categories
    parent = relationship("Category", remote_side=[id], backref="children")
    posts = relationship("Post", back_populates="category")
    
    def __repr__(self):
        return f"<Category(id={self.id}, name='{self.name}')>"


class Post(Base):
    """Post model with tags and comments."""
    __tablename__ = 'posts'
    
    id = Column(Integer, primary_key=True)
    title = Column(String(200), nullable=False, index=True)
    content = Column(String(5000))
    summary = Column(String(500))
    is_published = Column(Boolean, default=False, index=True)
    view_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    published_at = Column(DateTime)
    
    # Foreign keys
    author_id = Column(Integer, ForeignKey('users.id'), nullable=False, index=True)
    category_id = Column(Integer, ForeignKey('categories.id'))
    
    # Relationships
    author = relationship("User", back_populates="posts")
    category = relationship("Category", back_populates="posts")
    comments = relationship("Comment", back_populates="post", cascade="all, delete-orphan")
    tags = relationship("Tag", secondary="post_tags", back_populates="posts")
    
    def __repr__(self):
        return f"<Post(id={self.id}, title='{self.title}', author_id={self.author_id})>"
    
    def publish(self):
        """Publish the post."""
        self.is_published = True
        self.published_at = datetime.utcnow()
    
    def increment_view_count(self):
        """Increment view count."""
        self.view_count += 1


class Tag(Base):
    """Tag model for post categorization."""
    __tablename__ = 'tags'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(50), unique=True, nullable=False, index=True)
    color = Column(String(7))  # Hex color code
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationships
    posts = relationship("Post", secondary="post_tags", back_populates="tags")
    
    def __repr__(self):
        return f"<Tag(id={self.id}, name='{self.name}')>"


# Association table for many-to-many relationship between posts and tags
post_tags = Table(
    'post_tags',
    Base.metadata,
    Column('post_id', Integer, ForeignKey('posts.id'), primary_key=True),
    Column('tag_id', Integer, ForeignKey('tags.id'), primary_key=True)
)


class Comment(Base):
    """Comment model with hierarchical structure."""
    __tablename__ = 'comments'
    
    id = Column(Integer, primary_key=True)
    content = Column(String(1000), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_approved = Column(Boolean, default=True)
    
    # Foreign keys
    author_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    post_id = Column(Integer, ForeignKey('posts.id'), nullable=False, index=True)
    parent_id = Column(Integer, ForeignKey('comments.id'))
    
    # Relationships
    author = relationship("User", back_populates="comments")
    post = relationship("Post", back_populates="comments")
    parent = relationship("Comment", remote_side=[id], backref="replies")
    
    def __repr__(self):
        return f"<Comment(id={self.id}, author_id={self.author_id}, post_id={self.post_id})>"


class DatabaseManager:
    """Database manager with connection handling and utilities."""
    
    def __init__(self, database_url: str = "sqlite:///blog.db"):
        self.database_url = database_url
        self.engine = create_engine(
            database_url,
            echo=False,  # Set to True for SQL debugging
            poolclass=StaticPool if "sqlite" in database_url else None,
            connect_args={"check_same_thread": False} if "sqlite" in database_url else {}
        )
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
    
    def create_tables(self):
        """Create all tables."""
        Base.metadata.create_all(bind=self.engine)
        logger.info("Database tables created")
    
    def drop_tables(self):
        """Drop all tables."""
        Base.metadata.drop_all(bind=self.engine)
        logger.info("Database tables dropped")
    
    @contextlib.contextmanager
    def get_session(self):
        """Context manager for database sessions."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            session.close()
    
    def reset_database(self):
        """Reset database by dropping and recreating tables."""
        self.drop_tables()
        self.create_tables()


class UserRepository:
    """Repository for user-related database operations."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    def create_user(self, username: str, email: str, full_name: str = None) -> User:
        """Create a new user."""
        with self.db_manager.get_session() as session:
            user = User(username=username, email=email, full_name=full_name)
            session.add(user)
            session.flush()  # To get the ID
            session.refresh(user)
            return user
    
    def get_user_by_id(self, user_id: int) -> Optional[User]:
        """Get user by ID with all relationships loaded."""
        with self.db_manager.get_session() as session:
            return session.query(User).options(
                joinedload(User.profile),
                joinedload(User.roles)
            ).filter(User.id == user_id).first()
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username."""
        with self.db_manager.get_session() as session:
            return session.query(User).filter(User.username == username).first()
    
    def get_active_users(self) -> List[User]:
        """Get all active users."""
        with self.db_manager.get_session() as session:
            return session.query(User).filter(User.is_active == True).all()
    
    def search_users(self, query: str) -> List[User]:
        """Search users by username or email."""
        with self.db_manager.get_session() as session:
            return session.query(User).filter(
                or_(
                    User.username.contains(query),
                    User.email.contains(query),
                    User.full_name.contains(query)
                )
            ).all()
    
    def update_user(self, user_id: int, **kwargs) -> Optional[User]:
        """Update user."""
        with self.db_manager.get_session() as session:
            user = session.query(User).filter(User.id == user_id).first()
            if user:
                for key, value in kwargs.items():
                    if hasattr(user, key):
                        setattr(user, key, value)
                user.updated_at = datetime.utcnow()
                session.flush()
                session.refresh(user)
                return user
            return None
    
    def delete_user(self, user_id: int) -> bool:
        """Delete user."""
        with self.db_manager.get_session() as session:
            user = session.query(User).filter(User.id == user_id).first()
            if user:
                session.delete(user)
                return True
            return False
    
    def assign_role_to_user(self, user_id: int, role_name: str) -> bool:
        """Assign role to user."""
        with self.db_manager.get_session() as session:
            user = session.query(User).filter(User.id == user_id).first()
            role = session.query(Role).filter(Role.name == role_name).first()
            
            if user and role and role not in user.roles:
                user.roles.append(role)
                return True
            return False


class PostRepository:
    """Repository for post-related database operations."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    def create_post(self, title: str, content: str, author_id: int, 
                   category_id: int = None, tags: List[str] = None) -> Post:
        """Create a new post with tags."""
        with self.db_manager.get_session() as session:
            post = Post(
                title=title,
                content=content,
                author_id=author_id,
                category_id=category_id,
                summary=content[:200] + "..." if len(content) > 200 else content
            )
            
            # Add tags
            if tags:
                for tag_name in tags:
                    tag = session.query(Tag).filter(Tag.name == tag_name).first()
                    if not tag:
                        tag = Tag(name=tag_name)
                        session.add(tag)
                    post.tags.append(tag)
            
            session.add(post)
            session.flush()
            session.refresh(post)
            return post
    
    def get_published_posts(self, limit: int = 10, offset: int = 0) -> List[Post]:
        """Get published posts with pagination."""
        with self.db_manager.get_session() as session:
            return session.query(Post).options(
                joinedload(Post.author),
                joinedload(Post.category),
                joinedload(Post.tags)
            ).filter(Post.is_published == True).order_by(
                desc(Post.published_at)
            ).offset(offset).limit(limit).all()
    
    def get_posts_by_author(self, author_id: int) -> List[Post]:
        """Get all posts by a specific author."""
        with self.db_manager.get_session() as session:
            return session.query(Post).filter(Post.author_id == author_id).all()
    
    def get_posts_by_category(self, category_id: int) -> List[Post]:
        """Get posts by category."""
        with self.db_manager.get_session() as session:
            return session.query(Post).filter(Post.category_id == category_id).all()
    
    def get_posts_by_tag(self, tag_name: str) -> List[Post]:
        """Get posts by tag."""
        with self.db_manager.get_session() as session:
            return session.query(Post).join(Post.tags).filter(Tag.name == tag_name).all()
    
    def search_posts(self, query: str) -> List[Post]:
        """Search posts by title or content."""
        with self.db_manager.get_session() as session:
            return session.query(Post).filter(
                or_(
                    Post.title.contains(query),
                    Post.content.contains(query)
                )
            ).all()
    
    def get_popular_posts(self, limit: int = 10) -> List[Post]:
        """Get most viewed posts."""
        with self.db_manager.get_session() as session:
            return session.query(Post).filter(
                Post.is_published == True
            ).order_by(desc(Post.view_count)).limit(limit).all()
    
    def get_recent_posts(self, days: int = 7) -> List[Post]:
        """Get posts from the last N days."""
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        with self.db_manager.get_session() as session:
            return session.query(Post).filter(
                and_(
                    Post.is_published == True,
                    Post.published_at >= cutoff_date
                )
            ).order_by(desc(Post.published_at)).all()


class AnalyticsRepository:
    """Repository for analytics and reporting queries."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
    
    def get_user_stats(self) -> Dict[str, Any]:
        """Get user statistics."""
        with self.db_manager.get_session() as session:
            total_users = session.query(func.count(User.id)).scalar()
            active_users = session.query(func.count(User.id)).filter(User.is_active == True).scalar()
            users_with_posts = session.query(func.count(func.distinct(Post.author_id))).scalar()
            
            return {
                'total_users': total_users,
                'active_users': active_users,
                'inactive_users': total_users - active_users,
                'users_with_posts': users_with_posts
            }
    
    def get_post_stats(self) -> Dict[str, Any]:
        """Get post statistics."""
        with self.db_manager.get_session() as session:
            total_posts = session.query(func.count(Post.id)).scalar()
            published_posts = session.query(func.count(Post.id)).filter(Post.is_published == True).scalar()
            total_views = session.query(func.sum(Post.view_count)).scalar() or 0
            avg_views = session.query(func.avg(Post.view_count)).scalar() or 0
            
            return {
                'total_posts': total_posts,
                'published_posts': published_posts,
                'draft_posts': total_posts - published_posts,
                'total_views': total_views,
                'average_views_per_post': round(avg_views, 2)
            }
    
    def get_top_authors(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get top authors by post count."""
        with self.db_manager.get_session() as session:
            results = session.query(
                User.username,
                User.full_name,
                func.count(Post.id).label('post_count'),
                func.sum(Post.view_count).label('total_views')
            ).join(Post, User.id == Post.author_id).group_by(
                User.id, User.username, User.full_name
            ).order_by(desc('post_count')).limit(limit).all()
            
            return [
                {
                    'username': result.username,
                    'full_name': result.full_name,
                    'post_count': result.post_count,
                    'total_views': result.total_views or 0
                }
                for result in results
            ]
    
    def get_popular_tags(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most used tags."""
        with self.db_manager.get_session() as session:
            results = session.query(
                Tag.name,
                func.count(post_tags.c.post_id).label('usage_count')
            ).join(post_tags, Tag.id == post_tags.c.tag_id).group_by(
                Tag.id, Tag.name
            ).order_by(desc('usage_count')).limit(limit).all()
            
            return [
                {
                    'tag_name': result.name,
                    'usage_count': result.usage_count
                }
                for result in results
            ]


def seed_sample_data(db_manager: DatabaseManager):
    """Seed the database with sample data."""
    user_repo = UserRepository(db_manager)
    post_repo = PostRepository(db_manager)
    
    with db_manager.get_session() as session:
        # Create roles
        roles = [
            Role(name="admin", description="Administrator"),
            Role(name="author", description="Content Author"),
            Role(name="reader", description="Regular Reader")
        ]
        session.add_all(roles)
        session.flush()
        
        # Create categories
        categories = [
            Category(name="Technology", description="Tech-related posts"),
            Category(name="Programming", description="Programming tutorials"),
            Category(name="AI/ML", description="Artificial Intelligence and Machine Learning")
        ]
        session.add_all(categories)
        session.flush()
        
        # Create tags
        tags = [
            Tag(name="python", color="#3776ab"),
            Tag(name="javascript", color="#f7df1e"),
            Tag(name="machine-learning", color="#ff6f00"),
            Tag(name="tutorial", color="#4caf50"),
            Tag(name="beginner", color="#2196f3")
        ]
        session.add_all(tags)
        session.flush()
    
    # Create users
    users = []
    for i in range(1, 6):
        user = user_repo.create_user(
            username=f"user{i}",
            email=f"user{i}@example.com",
            full_name=f"User {i} Full Name"
        )
        users.append(user)
        
        # Assign roles
        role_name = "admin" if i == 1 else "author" if i <= 3 else "reader"
        user_repo.assign_role_to_user(user.id, role_name)
        
        # Create user profile
        with db_manager.get_session() as session:
            profile = UserProfile(
                user_id=user.id,
                bio=f"Bio for user {i}",
                location=f"City {i}"
            )
            session.add(profile)
    
    # Create posts
    sample_posts = [
        ("Introduction to Python", "Python is a powerful programming language...", ["python", "tutorial", "beginner"]),
        ("JavaScript Async/Await", "Modern JavaScript asynchronous programming...", ["javascript", "tutorial"]),
        ("Machine Learning Basics", "Getting started with ML algorithms...", ["machine-learning", "python"]),
        ("Advanced Python Features", "Exploring advanced Python concepts...", ["python"]),
        ("Web Development Tips", "Best practices for web development...", ["javascript", "tutorial"])
    ]
    
    for i, (title, content, tag_names) in enumerate(sample_posts):
        post = post_repo.create_post(
            title=title,
            content=content * 10,  # Make content longer
            author_id=users[i % len(users)].id,
            category_id=1,  # Technology category
            tags=tag_names
        )
        
        # Publish some posts
        if i < 3:
            with db_manager.get_session() as session:
                post_obj = session.query(Post).filter(Post.id == post.id).first()
                post_obj.publish()
                post_obj.view_count = (i + 1) * 10
        
        # Add comments
        with db_manager.get_session() as session:
            for j in range(2):
                comment = Comment(
                    content=f"Great post! Comment {j+1}",
                    author_id=users[(i + j + 1) % len(users)].id,
                    post_id=post.id
                )
                session.add(comment)
    
    logger.info("Sample data seeded successfully")


def demonstrate_basic_operations():
    """Demonstrate basic CRUD operations."""
    print("=== Basic Database Operations ===")
    
    # Initialize database
    db_manager = DatabaseManager("sqlite:///demo_blog.db")
    db_manager.reset_database()
    seed_sample_data(db_manager)
    
    # User operations
    user_repo = UserRepository(db_manager)
    
    # Get user by ID
    user = user_repo.get_user_by_id(1)
    print(f"User 1: {user.username} ({user.email})")
    print(f"Profile: {user.profile.bio if user.profile else 'No profile'}")
    print(f"Roles: {[role.name for role in user.roles]}")
    
    # Search users
    search_results = user_repo.search_users("user")
    print(f"Found {len(search_results)} users matching 'user'")
    
    return db_manager


def demonstrate_advanced_queries():
    """Demonstrate advanced database queries."""
    print("\n=== Advanced Database Queries ===")
    
    db_manager = DatabaseManager("sqlite:///demo_blog.db")
    post_repo = PostRepository(db_manager)
    analytics_repo = AnalyticsRepository(db_manager)
    
    # Get published posts
    published_posts = post_repo.get_published_posts(limit=5)
    print(f"Published posts: {len(published_posts)}")
    for post in published_posts:
        print(f"  - {post.title} by {post.author.username} ({post.view_count} views)")
    
    # Get popular posts
    popular_posts = post_repo.get_popular_posts(limit=3)
    print(f"\nTop 3 popular posts:")
    for post in popular_posts:
        print(f"  - {post.title}: {post.view_count} views")
    
    # Analytics
    user_stats = analytics_repo.get_user_stats()
    print(f"\nUser Statistics: {user_stats}")
    
    post_stats = analytics_repo.get_post_stats()
    print(f"Post Statistics: {post_stats}")
    
    top_authors = analytics_repo.get_top_authors()
    print(f"\nTop Authors:")
    for author in top_authors:
        print(f"  - {author['username']}: {author['post_count']} posts")
    
    popular_tags = analytics_repo.get_popular_tags()
    print(f"\nPopular Tags:")
    for tag in popular_tags:
        print(f"  - {tag['tag_name']}: {tag['usage_count']} uses")


def demonstrate_raw_sql():
    """Demonstrate raw SQL queries."""
    print("\n=== Raw SQL Queries ===")
    
    db_manager = DatabaseManager("sqlite:///demo_blog.db")
    
    with db_manager.get_session() as session:
        # Raw SQL query
        result = session.execute(text("""
            SELECT u.username, COUNT(p.id) as post_count, AVG(p.view_count) as avg_views
            FROM users u
            LEFT JOIN posts p ON u.id = p.author_id
            WHERE u.is_active = 1
            GROUP BY u.id, u.username
            HAVING post_count > 0
            ORDER BY post_count DESC
        """))
        
        print("Authors with their post statistics:")
        for row in result:
            print(f"  - {row.username}: {row.post_count} posts, {row.avg_views:.1f} avg views")


def demonstrate_relationships():
    """Demonstrate relationship loading and queries."""
    print("\n=== Relationship Demonstrations ===")
    
    db_manager = DatabaseManager("sqlite:///demo_blog.db")
    
    with db_manager.get_session() as session:
        # Eager loading with joinedload
        user = session.query(User).options(
            joinedload(User.posts).joinedload(Post.tags),
            joinedload(User.posts).joinedload(Post.comments),
            joinedload(User.profile),
            joinedload(User.roles)
        ).filter(User.id == 1).first()
        
        print(f"User: {user.username}")
        print(f"Posts: {len(user.posts)}")
        for post in user.posts:
            print(f"  - {post.title}")
            print(f"    Tags: {[tag.name for tag in post.tags]}")
            print(f"    Comments: {len(post.comments)}")
        
        # Hierarchical data (categories)
        categories = session.query(Category).filter(Category.parent_id.is_(None)).all()
        print(f"\nTop-level categories: {len(categories)}")
        for category in categories:
            print(f"  - {category.name}: {len(category.posts)} posts")


if __name__ == "__main__":
    print("=== Database and ORM Examples ===\n")
    
    # Run demonstrations
    db_manager = demonstrate_basic_operations()
    demonstrate_advanced_queries()
    demonstrate_raw_sql()
    demonstrate_relationships()
    
    print("\n=== Database Features Demonstrated ===")
    print("- SQLAlchemy ORM models with relationships")
    print("- One-to-one, one-to-many, and many-to-many relationships")
    print("- Repository pattern for data access")
    print("- Advanced queries with joins and aggregations")
    print("- Eager loading with joinedload")
    print("- Raw SQL execution")
    print("- Database connection management")
    print("- Transaction handling")
    print("- Analytics and reporting queries")
    print("- Hierarchical data structures")
    print("- Data seeding and migrations")