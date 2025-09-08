# Basic Django Dataset - Python Web Framework Fundamentals

## Dataset 1: Django Project Setup and Basic Views
```python
# myproject/settings.py
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = 'your-secret-key-here'
DEBUG = True
ALLOWED_HOSTS = ['localhost', '127.0.0.1']

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'blog',  # Custom app
    'accounts',  # Custom app
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'myproject.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [BASE_DIR / 'templates'],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

STATIC_URL = '/static/'
STATICFILES_DIRS = [BASE_DIR / 'static']
STATIC_ROOT = BASE_DIR / 'staticfiles'

MEDIA_URL = '/media/'
MEDIA_ROOT = BASE_DIR / 'media'

# myproject/urls.py
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static
from . import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.home, name='home'),
    path('about/', views.about, name='about'),
    path('blog/', include('blog.urls')),
    path('accounts/', include('accounts.urls')),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

# myproject/views.py
from django.shortcuts import render
from django.http import HttpResponse
from django.contrib.auth.decorators import login_required
from blog.models import Post

def home(request):
    """Home page view with recent blog posts"""
    recent_posts = Post.objects.filter(published=True).order_by('-created_at')[:5]
    context = {
        'recent_posts': recent_posts,
        'page_title': 'Welcome to My Blog'
    }
    return render(request, 'home.html', context)

def about(request):
    """About page view"""
    context = {
        'page_title': 'About Us',
        'description': 'This is a Django blog application demonstrating basic concepts.'
    }
    return render(request, 'about.html', context)

@login_required
def dashboard(request):
    """User dashboard - requires login"""
    user_posts = Post.objects.filter(author=request.user)
    context = {
        'user_posts': user_posts,
        'total_posts': user_posts.count(),
        'page_title': 'Dashboard'
    }
    return render(request, 'dashboard.html', context)

# Basic function-based views
def hello_world(request):
    return HttpResponse("Hello, Django World!")

def hello_user(request, username):
    return HttpResponse(f"Hello, {username}!")

def user_profile(request, user_id):
    try:
        user_id = int(user_id)
        return HttpResponse(f"Profile for user ID: {user_id}")
    except ValueError:
        return HttpResponse("Invalid user ID", status=400)
```

## Dataset 2: Django Models and Database Operations
```python
# blog/models.py
from django.db import models
from django.contrib.auth.models import User
from django.urls import reverse
from django.utils import timezone
from django.utils.text import slugify

class Category(models.Model):
    name = models.CharField(max_length=100, unique=True)
    slug = models.SlugField(max_length=100, unique=True, blank=True)
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        verbose_name_plural = "Categories"
        ordering = ['name']
    
    def __str__(self):
        return self.name
    
    def save(self, *args, **kwargs):
        if not self.slug:
            self.slug = slugify(self.name)
        super().save(*args, **kwargs)
    
    def get_absolute_url(self):
        return reverse('blog:category', kwargs={'slug': self.slug})

class Tag(models.Model):
    name = models.CharField(max_length=50, unique=True)
    slug = models.SlugField(max_length=50, unique=True, blank=True)
    
    def __str__(self):
        return self.name
    
    def save(self, *args, **kwargs):
        if not self.slug:
            self.slug = slugify(self.name)
        super().save(*args, **kwargs)

class PostManager(models.Manager):
    def published(self):
        return self.filter(published=True, publish_date__lte=timezone.now())
    
    def by_author(self, author):
        return self.filter(author=author)

class Post(models.Model):
    STATUS_CHOICES = [
        ('draft', 'Draft'),
        ('published', 'Published'),
        ('archived', 'Archived'),
    ]
    
    title = models.CharField(max_length=200)
    slug = models.SlugField(max_length=200, unique=True, blank=True)
    author = models.ForeignKey(User, on_delete=models.CASCADE, related_name='posts')
    category = models.ForeignKey(Category, on_delete=models.SET_NULL, null=True, blank=True)
    tags = models.ManyToManyField(Tag, blank=True)
    
    content = models.TextField()
    excerpt = models.CharField(max_length=300, blank=True)
    
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='draft')
    published = models.BooleanField(default=False)
    featured = models.BooleanField(default=False)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    publish_date = models.DateTimeField(default=timezone.now)
    
    views_count = models.PositiveIntegerField(default=0)
    likes_count = models.PositiveIntegerField(default=0)
    
    # Custom manager
    objects = PostManager()
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['-created_at']),
            models.Index(fields=['published', '-publish_date']),
        ]
    
    def __str__(self):
        return self.title
    
    def save(self, *args, **kwargs):
        if not self.slug:
            self.slug = slugify(self.title)
        
        # Auto-generate excerpt if not provided
        if not self.excerpt:
            self.excerpt = self.content[:297] + '...' if len(self.content) > 300 else self.content
        
        super().save(*args, **kwargs)
    
    def get_absolute_url(self):
        return reverse('blog:post_detail', kwargs={
            'year': self.publish_date.year,
            'month': self.publish_date.month,
            'day': self.publish_date.day,
            'slug': self.slug
        })
    
    def increment_views(self):
        self.views_count += 1
        self.save(update_fields=['views_count'])
    
    def is_published(self):
        return self.published and self.publish_date <= timezone.now()
    
    @property
    def reading_time(self):
        """Estimate reading time in minutes"""
        words_per_minute = 200
        word_count = len(self.content.split())
        return max(1, round(word_count / words_per_minute))

class Comment(models.Model):
    post = models.ForeignKey(Post, on_delete=models.CASCADE, related_name='comments')
    author = models.ForeignKey(User, on_delete=models.CASCADE)
    parent = models.ForeignKey('self', on_delete=models.CASCADE, null=True, blank=True, related_name='replies')
    
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    approved = models.BooleanField(default=True)
    
    class Meta:
        ordering = ['created_at']
    
    def __str__(self):
        return f'Comment by {self.author.username} on {self.post.title}'
    
    def is_reply(self):
        return self.parent is not None

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    bio = models.TextField(max_length=500, blank=True)
    location = models.CharField(max_length=30, blank=True)
    birth_date = models.DateField(null=True, blank=True)
    avatar = models.ImageField(upload_to='avatars/', null=True, blank=True)
    website = models.URLField(blank=True)
    
    # Social media links
    twitter = models.CharField(max_length=50, blank=True)
    github = models.CharField(max_length=50, blank=True)
    linkedin = models.CharField(max_length=50, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f'{self.user.username} Profile'
    
    def get_absolute_url(self):
        return reverse('accounts:profile', kwargs={'username': self.user.username})
    
    @property
    def full_name(self):
        return f'{self.user.first_name} {self.user.last_name}'.strip() or self.user.username

# Database operations examples
def example_queries():
    # Creating objects
    category = Category.objects.create(
        name='Technology',
        description='Posts about technology and programming'
    )
    
    # Getting or creating objects
    tag, created = Tag.objects.get_or_create(
        name='Django',
        defaults={'slug': 'django'}
    )
    
    # Complex queries
    from django.db.models import Q, Count, Avg
    
    # Posts with specific conditions
    recent_posts = Post.objects.published().filter(
        Q(category__name='Technology') | Q(tags__name='Django')
    ).distinct()
    
    # Aggregation
    post_stats = Post.objects.aggregate(
        total_posts=Count('id'),
        avg_views=Avg('views_count'),
        total_views=models.Sum('views_count')
    )
    
    # Annotations
    authors_with_post_count = User.objects.annotate(
        post_count=Count('posts'),
        published_count=Count('posts', filter=Q(posts__published=True))
    ).filter(post_count__gt=0)
    
    # Prefetch related for optimization
    posts_with_comments = Post.objects.published().prefetch_related(
        'comments__author',
        'tags',
        'category'
    ).select_related('author')
    
    return {
        'recent_posts': recent_posts,
        'post_stats': post_stats,
        'authors_with_post_count': authors_with_post_count,
        'posts_with_comments': posts_with_comments
    }
```

## Dataset 3: Django Templates and Template Tags
```html
<!-- templates/base.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}{% endblock %} - My Django Blog</title>
    
    {% load static %}
    <link rel="stylesheet" href="{% static 'css/style.css' %}">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.1.3/css/bootstrap.min.css">
    
    {% block extra_css %}{% endblock %}
</head>
<body>
    <!-- Navigation -->
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
        <div class="container">
            <a class="navbar-brand" href="{% url 'home' %}">My Blog</a>
            
            <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarNav">
                <span class="navbar-toggler-icon"></span>
            </button>
            
            <div class="collapse navbar-collapse" id="navbarNav">
                <ul class="navbar-nav me-auto">
                    <li class="nav-item">
                        <a class="nav-link {% if request.resolver_match.url_name == 'home' %}active{% endif %}" 
                           href="{% url 'home' %}">Home</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link {% if request.resolver_match.url_name == 'about' %}active{% endif %}" 
                           href="{% url 'about' %}">About</a>
                    </li>
                    <li class="nav-item">
                        <a class="nav-link" href="{% url 'blog:post_list' %}">Blog</a>
                    </li>
                </ul>
                
                <ul class="navbar-nav">
                    {% if user.is_authenticated %}
                        <li class="nav-item dropdown">
                            <a class="nav-link dropdown-toggle" href="#" id="navbarDropdown" 
                               role="button" data-bs-toggle="dropdown">
                                {{ user.username }}
                            </a>
                            <ul class="dropdown-menu">
                                <li><a class="dropdown-item" href="{% url 'accounts:profile' user.username %}">Profile</a></li>
                                <li><a class="dropdown-item" href="{% url 'dashboard' %}">Dashboard</a></li>
                                <li><hr class="dropdown-divider"></li>
                                <li><a class="dropdown-item" href="{% url 'accounts:logout' %}">Logout</a></li>
                            </ul>
                        </li>
                    {% else %}
                        <li class="nav-item">
                            <a class="nav-link" href="{% url 'accounts:login' %}">Login</a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" href="{% url 'accounts:register' %}">Register</a>
                        </li>
                    {% endif %}
                </ul>
            </div>
        </div>
    </nav>
    
    <!-- Messages -->
    {% if messages %}
        <div class="container mt-3">
            {% for message in messages %}
                <div class="alert alert-{{ message.tags }} alert-dismissible fade show" role="alert">
                    {{ message }}
                    <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
                </div>
            {% endfor %}
        </div>
    {% endif %}
    
    <!-- Main content -->
    <main class="container my-4">
        {% block content %}{% endblock %}
    </main>
    
    <!-- Footer -->
    <footer class="bg-dark text-light py-4 mt-5">
        <div class="container">
            <div class="row">
                <div class="col-md-6">
                    <h5>My Django Blog</h5>
                    <p>A blog built with Django demonstrating best practices.</p>
                </div>
                <div class="col-md-6">
                    <h5>Quick Links</h5>
                    <ul class="list-unstyled">
                        <li><a href="{% url 'home' %}" class="text-light">Home</a></li>
                        <li><a href="{% url 'blog:post_list' %}" class="text-light">All Posts</a></li>
                        <li><a href="{% url 'about' %}" class="text-light">About</a></li>
                    </ul>
                </div>
            </div>
            <hr>
            <div class="text-center">
                <p>&copy; {% now "Y" %} My Django Blog. All rights reserved.</p>
            </div>
        </div>
    </footer>
    
    <script src="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.1.3/js/bootstrap.bundle.min.js"></script>
    {% block extra_js %}{% endblock %}
</body>
</html>

<!-- templates/home.html -->
{% extends 'base.html' %}
{% load static %}
{% load blog_extras %}

{% block title %}{{ page_title }}{% endblock %}

{% block content %}
<div class="row">
    <div class="col-lg-8">
        <!-- Hero section -->
        <div class="jumbotron bg-primary text-white p-4 rounded mb-4">
            <h1 class="display-4">Welcome to My Blog</h1>
            <p class="lead">Discover amazing content about technology, programming, and more.</p>
            <a class="btn btn-light btn-lg" href="{% url 'blog:post_list' %}" role="button">
                Read All Posts
            </a>
        </div>
        
        <!-- Recent posts -->
        {% if recent_posts %}
            <h2>Recent Posts</h2>
            <div class="row">
                {% for post in recent_posts %}
                    <div class="col-md-6 mb-4">
                        <div class="card h-100">
                            {% if post.featured %}
                                <div class="badge bg-warning position-absolute top-0 end-0 m-2">Featured</div>
                            {% endif %}
                            
                            <div class="card-body">
                                <h5 class="card-title">
                                    <a href="{{ post.get_absolute_url }}" class="text-decoration-none">
                                        {{ post.title }}
                                    </a>
                                </h5>
                                <p class="card-text">{{ post.excerpt }}</p>
                                
                                <div class="d-flex justify-content-between align-items-center">
                                    <small class="text-muted">
                                        By {{ post.author.get_full_name|default:post.author.username }}
                                    </small>
                                    <small class="text-muted">
                                        {{ post.publish_date|timesince }} ago
                                    </small>
                                </div>
                                
                                <!-- Tags -->
                                {% if post.tags.all %}
                                    <div class="mt-2">
                                        {% for tag in post.tags.all %}
                                            <span class="badge bg-secondary me-1">{{ tag.name }}</span>
                                        {% endfor %}
                                    </div>
                                {% endif %}
                                
                                <!-- Reading time -->
                                <div class="mt-2">
                                    <small class="text-muted">
                                        <i class="bi bi-clock"></i> {{ post.reading_time }} min read
                                    </small>
                                </div>
                            </div>
                            
                            <div class="card-footer">
                                <a href="{{ post.get_absolute_url }}" class="btn btn-primary btn-sm">
                                    Read More
                                </a>
                                <small class="text-muted float-end">
                                    {{ post.views_count }} views
                                </small>
                            </div>
                        </div>
                    </div>
                {% endfor %}
            </div>
        {% else %}
            <div class="alert alert-info">
                <h4>No posts yet!</h4>
                <p>Check back later for amazing content.</p>
            </div>
        {% endif %}
    </div>
    
    <!-- Sidebar -->
    <div class="col-lg-4">
        {% include 'partials/sidebar.html' %}
    </div>
</div>
{% endblock %}

<!-- templates/blog/post_detail.html -->
{% extends 'base.html' %}
{% load static %}
{% load blog_extras %}

{% block title %}{{ post.title }}{% endblock %}

{% block extra_css %}
<style>
    .post-meta {
        border-bottom: 1px solid #eee;
        padding-bottom: 1rem;
        margin-bottom: 2rem;
    }
    
    .post-content {
        line-height: 1.8;
        font-size: 1.1rem;
    }
    
    .comment-item {
        border-left: 3px solid #007bff;
        padding-left: 1rem;
        margin-bottom: 1rem;
    }
    
    .comment-reply {
        margin-left: 2rem;
        border-left-color: #6c757d;
    }
</style>
{% endblock %}

{% block content %}
<article class="post-detail">
    <!-- Post header -->
    <header class="post-header mb-4">
        <h1 class="post-title">{{ post.title }}</h1>
        
        <div class="post-meta">
            <div class="row align-items-center">
                <div class="col-md-8">
                    <div class="d-flex align-items-center">
                        {% if post.author.userprofile.avatar %}
                            <img src="{{ post.author.userprofile.avatar.url }}" 
                                 alt="{{ post.author.username }}" 
                                 class="rounded-circle me-3" 
                                 width="50" height="50">
                        {% endif %}
                        
                        <div>
                            <strong>
                                <a href="{% url 'accounts:profile' post.author.username %}" 
                                   class="text-decoration-none">
                                    {{ post.author.get_full_name|default:post.author.username }}
                                </a>
                            </strong>
                            <br>
                            <small class="text-muted">
                                Published {{ post.publish_date|date:"F d, Y" }}
                                {% if post.updated_at != post.created_at %}
                                    • Updated {{ post.updated_at|timesince }} ago
                                {% endif %}
                            </small>
                        </div>
                    </div>
                </div>
                
                <div class="col-md-4 text-md-end">
                    <small class="text-muted">
                        <i class="bi bi-eye"></i> {{ post.views_count }} views<br>
                        <i class="bi bi-clock"></i> {{ post.reading_time }} min read
                    </small>
                </div>
            </div>
            
            <!-- Category and tags -->
            <div class="mt-3">
                {% if post.category %}
                    <a href="{{ post.category.get_absolute_url }}" 
                       class="badge bg-primary text-decoration-none me-2">
                        {{ post.category.name }}
                    </a>
                {% endif %}
                
                {% for tag in post.tags.all %}
                    <span class="badge bg-secondary me-1">{{ tag.name }}</span>
                {% endfor %}
            </div>
        </div>
    </header>
    
    <!-- Post content -->
    <div class="post-content">
        {{ post.content|linebreaks }}
    </div>
    
    <!-- Navigation to next/previous posts -->
    <nav class="post-navigation mt-5 pt-4 border-top">
        <div class="row">
            {% get_previous_post post as previous_post %}
            {% get_next_post post as next_post %}
            
            <div class="col-md-6">
                {% if previous_post %}
                    <div class="nav-previous">
                        <small class="text-muted">Previous Post</small>
                        <br>
                        <a href="{{ previous_post.get_absolute_url }}" class="text-decoration-none">
                            ← {{ previous_post.title }}
                        </a>
                    </div>
                {% endif %}
            </div>
            
            <div class="col-md-6 text-md-end">
                {% if next_post %}
                    <div class="nav-next">
                        <small class="text-muted">Next Post</small>
                        <br>
                        <a href="{{ next_post.get_absolute_url }}" class="text-decoration-none">
                            {{ next_post.title }} →
                        </a>
                    </div>
                {% endif %}
            </div>
        </div>
    </nav>
</article>

<!-- Comments section -->
<section class="comments-section mt-5 pt-4 border-top">
    <h3>Comments ({{ post.comments.count }})</h3>
    
    {% if user.is_authenticated %}
        <!-- Comment form -->
        <form method="post" action="{% url 'blog:add_comment' post.pk %}" class="mb-4">
            {% csrf_token %}
            <div class="mb-3">
                <label for="comment-content" class="form-label">Leave a comment:</label>
                <textarea class="form-control" id="comment-content" name="content" 
                          rows="4" required></textarea>
            </div>
            <button type="submit" class="btn btn-primary">Post Comment</button>
        </form>
    {% else %}
        <p class="alert alert-info">
            <a href="{% url 'accounts:login' %}">Login</a> to leave a comment.
        </p>
    {% endif %}
    
    <!-- Comments list -->
    {% for comment in post.comments.all %}
        {% if not comment.parent %}
            <div class="comment-item">
                <div class="d-flex justify-content-between">
                    <strong>{{ comment.author.username }}</strong>
                    <small class="text-muted">{{ comment.created_at|timesince }} ago</small>
                </div>
                <p class="mt-2">{{ comment.content|linebreaks }}</p>
                
                <!-- Replies -->
                {% for reply in comment.replies.all %}
                    <div class="comment-item comment-reply">
                        <div class="d-flex justify-content-between">
                            <strong>{{ reply.author.username }}</strong>
                            <small class="text-muted">{{ reply.created_at|timesince }} ago</small>
                        </div>
                        <p class="mt-2">{{ reply.content|linebreaks }}</p>
                    </div>
                {% endfor %}
            </div>
        {% endif %}
    {% empty %}
        <p class="text-muted">No comments yet. Be the first to comment!</p>
    {% endfor %}
</section>
{% endblock %}

<!-- templates/partials/sidebar.html -->
<div class="sidebar">
    <!-- Search -->
    <div class="card mb-4">
        <div class="card-header">
            <h5 class="mb-0">Search</h5>
        </div>
        <div class="card-body">
            <form method="get" action="{% url 'blog:search' %}">
                <div class="input-group">
                    <input type="text" class="form-control" name="q" 
                           placeholder="Search posts..." value="{{ request.GET.q }}">
                    <button class="btn btn-outline-secondary" type="submit">
                        Search
                    </button>
                </div>
            </form>
        </div>
    </div>
    
    <!-- Recent posts -->
    {% get_recent_posts 5 as recent_posts %}
    {% if recent_posts %}
        <div class="card mb-4">
            <div class="card-header">
                <h5 class="mb-0">Recent Posts</h5>
            </div>
            <div class="card-body">
                {% for post in recent_posts %}
                    <div class="mb-3">
                        <h6 class="mb-1">
                            <a href="{{ post.get_absolute_url }}" class="text-decoration-none">
                                {{ post.title|truncatechars:50 }}
                            </a>
                        </h6>
                        <small class="text-muted">{{ post.publish_date|date:"M d, Y" }}</small>
                    </div>
                {% endfor %}
            </div>
        </div>
    {% endif %}
    
    <!-- Categories -->
    {% get_categories as categories %}
    {% if categories %}
        <div class="card mb-4">
            <div class="card-header">
                <h5 class="mb-0">Categories</h5>
            </div>
            <div class="card-body">
                {% for category in categories %}
                    <div class="d-flex justify-content-between mb-2">
                        <a href="{{ category.get_absolute_url }}" class="text-decoration-none">
                            {{ category.name }}
                        </a>
                        <span class="badge bg-secondary">{{ category.posts.count }}</span>
                    </div>
                {% endfor %}
            </div>
        </div>
    {% endif %}
    
    <!-- Popular tags -->
    {% get_popular_tags 10 as popular_tags %}
    {% if popular_tags %}
        <div class="card">
            <div class="card-header">
                <h5 class="mb-0">Popular Tags</h5>
            </div>
            <div class="card-body">
                {% for tag in popular_tags %}
                    <span class="badge bg-outline-secondary me-1 mb-1">{{ tag.name }}</span>
                {% endfor %}
            </div>
        </div>
    {% endif %}
</div>
```

## Dataset 4: Custom Template Tags and Filters
```python
# blog/templatetags/__init__.py
# Empty file to make this directory a Python package

# blog/templatetags/blog_extras.py
from django import template
from django.db.models import Count, Q
from django.utils.safestring import mark_safe
from django.utils.html import escape
from blog.models import Post, Category, Tag
import re

register = template.Library()

@register.simple_tag
def get_recent_posts(count=5):
    """Get recent published posts"""
    return Post.objects.published()[:count]

@register.simple_tag
def get_popular_posts(count=5):
    """Get most viewed posts"""
    return Post.objects.published().order_by('-views_count')[:count]

@register.simple_tag
def get_categories():
    """Get all categories with post count"""
    return Category.objects.annotate(
        post_count=Count('post', filter=Q(post__published=True))
    ).filter(post_count__gt=0)

@register.simple_tag
def get_popular_tags(count=10):
    """Get popular tags based on usage"""
    return Tag.objects.annotate(
        post_count=Count('post', filter=Q(post__published=True))
    ).filter(post_count__gt=0).order_by('-post_count')[:count]

@register.simple_tag
def get_archive_dates():
    """Get archive dates for posts"""
    from django.db.models.functions import TruncMonth
    return Post.objects.published().annotate(
        month=TruncMonth('publish_date')
    ).values('month').annotate(
        post_count=Count('id')
    ).order_by('-month')

@register.simple_tag
def get_related_posts(post, count=3):
    """Get related posts based on tags and category"""
    related = Post.objects.published().exclude(id=post.id)
    
    if post.tags.exists():
        related = related.filter(tags__in=post.tags.all()).distinct()
    elif post.category:
        related = related.filter(category=post.category)
    
    return related[:count]

@register.simple_tag(takes_context=True)
def get_previous_post(context, post):
    """Get previous post by publish date"""
    return Post.objects.published().filter(
        publish_date__lt=post.publish_date
    ).order_by('-publish_date').first()

@register.simple_tag(takes_context=True)
def get_next_post(context, post):
    """Get next post by publish date"""
    return Post.objects.published().filter(
        publish_date__gt=post.publish_date
    ).order_by('publish_date').first()

@register.inclusion_tag('blog/tags/post_card.html')
def post_card(post, show_excerpt=True):
    """Render a post card"""
    return {
        'post': post,
        'show_excerpt': show_excerpt
    }

@register.inclusion_tag('blog/tags/breadcrumb.html', takes_context=True)
def breadcrumb(context, *args):
    """Generate breadcrumb navigation"""
    request = context['request']
    breadcrumbs = [{'name': 'Home', 'url': '/'}]
    
    for item in args:
        if hasattr(item, 'get_absolute_url'):
            breadcrumbs.append({
                'name': str(item),
                'url': item.get_absolute_url()
            })
        else:
            breadcrumbs.append({'name': str(item), 'url': None})
    
    return {'breadcrumbs': breadcrumbs}

# Custom filters
@register.filter
def truncate_words_html(value, arg):
    """
    Truncate HTML content to specified number of words
    while preserving HTML tags
    """
    try:
        length = int(arg)
    except ValueError:
        return value
    
    words = value.split()
    if len(words) <= length:
        return value
    
    truncated = ' '.join(words[:length])
    return mark_safe(f"{truncated}...")

@register.filter
def reading_time(content):
    """Calculate estimated reading time"""
    words_per_minute = 200
    word_count = len(content.split())
    time = max(1, round(word_count / words_per_minute))
    return f"{time} min read"

@register.filter
def add_class(field, css_class):
    """Add CSS class to form field"""
    return field.as_widget(attrs={'class': css_class})

@register.filter
def get_item(dictionary, key):
    """Get item from dictionary by key"""
    return dictionary.get(key)

@register.filter
def multiply(value, arg):
    """Multiply value by argument"""
    try:
        return float(value) * float(arg)
    except (ValueError, TypeError):
        return 0

@register.filter
def percentage(value, total):
    """Calculate percentage"""
    try:
        return round((float(value) / float(total)) * 100, 1)
    except (ValueError, TypeError, ZeroDivisionError):
        return 0

@register.filter
def highlight_search(text, search_term):
    """Highlight search terms in text"""
    if not search_term:
        return text
    
    highlighted = re.sub(
        f'({re.escape(search_term)})',
        r'<mark>\1</mark>',
        escape(text),
        flags=re.IGNORECASE
    )
    return mark_safe(highlighted)

@register.filter
def social_share_url(post, platform):
    """Generate social media share URLs"""
    post_url = post.get_absolute_url()
    post_title = post.title
    
    urls = {
        'twitter': f'https://twitter.com/intent/tweet?text={post_title}&url={post_url}',
        'facebook': f'https://www.facebook.com/sharer/sharer.php?u={post_url}',
        'linkedin': f'https://www.linkedin.com/sharing/share-offsite/?url={post_url}',
        'reddit': f'https://reddit.com/submit?url={post_url}&title={post_title}',
    }
    
    return urls.get(platform, '')

# Assignment tags (deprecated but showing alternative)
@register.simple_tag
def query_string(request, **kwargs):
    """
    Generate query string from current request parameters
    and additional parameters
    """
    import urllib.parse
    
    query_dict = request.GET.copy()
    for key, value in kwargs.items():
        if value is None:
            query_dict.pop(key, None)
        else:
            query_dict[key] = value
    
    query_string = urllib.parse.urlencode(query_dict)
    return f'?{query_string}' if query_string else ''

# Custom filter for pagination
@register.filter
def pagination_window(page, window_size=3):
    """Generate pagination window around current page"""
    current_page = page.number
    total_pages = page.paginator.num_pages
    
    start = max(1, current_page - window_size)
    end = min(total_pages + 1, current_page + window_size + 1)
    
    return range(start, end)

# Usage examples in templates:
"""
{% load blog_extras %}

<!-- Simple tags -->
{% get_recent_posts 5 as recent_posts %}
{% get_popular_tags 10 as tags %}
{% get_related_posts post 3 as related %}

<!-- Inclusion tags -->
{% post_card post %}
{% breadcrumb category post %}

<!-- Filters -->
{{ post.content|truncate_words_html:50 }}
{{ post.content|reading_time }}
{{ form.email|add_class:"form-control" }}
{{ search_query|highlight_search:request.GET.q }}

<!-- Social sharing -->
<a href="{{ post|social_share_url:'twitter' }}">Share on Twitter</a>

<!-- Pagination -->
{% for page_num in page_obj|pagination_window:2 %}
    <a href="?page={{ page_num }}">{{ page_num }}</a>
{% endfor %}
"""
```