# Basic Express.js Dataset - Node.js Web Framework Fundamentals

## Dataset 1: Express Server Setup and Basic Routes
```javascript
// package.json
{
  "name": "express-blog-api",
  "version": "1.0.0",
  "description": "A basic blog API built with Express.js",
  "main": "server.js",
  "scripts": {
    "start": "node server.js",
    "dev": "nodemon server.js",
    "test": "jest"
  },
  "dependencies": {
    "express": "^4.18.2",
    "mongoose": "^7.0.3",
    "bcryptjs": "^2.4.3",
    "jsonwebtoken": "^9.0.0",
    "cors": "^2.8.5",
    "helmet": "^6.1.5",
    "morgan": "^1.10.0",
    "express-rate-limit": "^6.7.0",
    "express-validator": "^6.15.0",
    "multer": "^1.4.5-lts.1",
    "dotenv": "^16.0.3"
  },
  "devDependencies": {
    "nodemon": "^2.0.22",
    "jest": "^29.5.0",
    "supertest": "^6.3.3"
  },
  "engines": {
    "node": ">=16.0.0"
  }
}

// server.js - Main application file
const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const morgan = require('morgan');
const rateLimit = require('express-rate-limit');
require('dotenv').config();

const connectDB = require('./config/database');
const authRoutes = require('./routes/auth');
const postRoutes = require('./routes/posts');
const userRoutes = require('./routes/users');
const errorHandler = require('./middleware/errorHandler');
const notFound = require('./middleware/notFound');

const app = express();
const PORT = process.env.PORT || 3000;

// Connect to database
connectDB();

// Security middleware
app.use(helmet());
app.use(cors({
    origin: process.env.CLIENT_URL || 'http://localhost:3000',
    credentials: true
}));

// Rate limiting
const limiter = rateLimit({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 100 // limit each IP to 100 requests per windowMs
});
app.use(limiter);

// Logging
app.use(morgan('combined'));

// Body parsing middleware
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true }));

// Static files
app.use('/uploads', express.static('uploads'));

// Basic routes
app.get('/', (req, res) => {
    res.json({
        message: 'Welcome to Express Blog API',
        version: '1.0.0',
        endpoints: {
            auth: '/api/auth',
            posts: '/api/posts',
            users: '/api/users'
        }
    });
});

app.get('/health', (req, res) => {
    res.status(200).json({
        status: 'OK',
        timestamp: new Date().toISOString(),
        uptime: process.uptime()
    });
});

// API routes
app.use('/api/auth', authRoutes);
app.use('/api/posts', postRoutes);
app.use('/api/users', userRoutes);

// Error handling middleware
app.use(notFound);
app.use(errorHandler);

// Start server
app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
    console.log(`Environment: ${process.env.NODE_ENV || 'development'}`);
});

module.exports = app; // For testing

// config/database.js - Database connection
const mongoose = require('mongoose');

const connectDB = async () => {
    try {
        const conn = await mongoose.connect(process.env.MONGODB_URI || 'mongodb://localhost:27017/express-blog', {
            useNewUrlParser: true,
            useUnifiedTopology: true,
        });

        console.log(`MongoDB Connected: ${conn.connection.host}`);
        
        // Connection event listeners
        mongoose.connection.on('error', (err) => {
            console.error('MongoDB connection error:', err);
        });

        mongoose.connection.on('disconnected', () => {
            console.log('MongoDB disconnected');
        });

        // Graceful shutdown
        process.on('SIGINT', async () => {
            await mongoose.connection.close();
            console.log('MongoDB connection closed through app termination');
            process.exit(0);
        });

    } catch (error) {
        console.error('Database connection failed:', error);
        process.exit(1);
    }
};

module.exports = connectDB;

// routes/posts.js - Basic CRUD routes
const express = require('express');
const router = express.Router();
const { body, validationResult } = require('express-validator');
const Post = require('../models/Post');
const auth = require('../middleware/auth');

// @route   GET /api/posts
// @desc    Get all posts with pagination and filtering
// @access  Public
router.get('/', async (req, res) => {
    try {
        const {
            page = 1,
            limit = 10,
            search,
            category,
            author,
            status = 'published'
        } = req.query;

        // Build query
        const query = { status };

        if (search) {
            query.$or = [
                { title: { $regex: search, $options: 'i' } },
                { content: { $regex: search, $options: 'i' } }
            ];
        }

        if (category) {
            query.category = category;
        }

        if (author) {
            query.author = author;
        }

        // Execute query with pagination
        const posts = await Post.find(query)
            .populate('author', 'name email')
            .populate('category', 'name slug')
            .sort({ createdAt: -1 })
            .limit(limit * 1)
            .skip((page - 1) * limit)
            .lean();

        const total = await Post.countDocuments(query);

        res.json({
            posts,
            currentPage: parseInt(page),
            totalPages: Math.ceil(total / limit),
            totalPosts: total,
            hasNext: page < Math.ceil(total / limit),
            hasPrev: page > 1
        });

    } catch (error) {
        console.error('Error fetching posts:', error);
        res.status(500).json({ message: 'Server error' });
    }
});

// @route   GET /api/posts/:id
// @desc    Get single post by ID
// @access  Public
router.get('/:id', async (req, res) => {
    try {
        const post = await Post.findById(req.params.id)
            .populate('author', 'name email avatar')
            .populate('category', 'name slug')
            .populate('comments.author', 'name email');

        if (!post) {
            return res.status(404).json({ message: 'Post not found' });
        }

        // Increment view count
        post.views += 1;
        await post.save();

        res.json(post);

    } catch (error) {
        console.error('Error fetching post:', error);
        
        if (error.name === 'CastError') {
            return res.status(400).json({ message: 'Invalid post ID' });
        }
        
        res.status(500).json({ message: 'Server error' });
    }
});

// @route   POST /api/posts
// @desc    Create new post
// @access  Private
router.post('/', [
    auth,
    body('title')
        .trim()
        .isLength({ min: 5, max: 200 })
        .withMessage('Title must be between 5 and 200 characters'),
    body('content')
        .trim()
        .isLength({ min: 50 })
        .withMessage('Content must be at least 50 characters'),
    body('excerpt')
        .optional()
        .trim()
        .isLength({ max: 300 })
        .withMessage('Excerpt must not exceed 300 characters'),
    body('category')
        .optional()
        .isMongoId()
        .withMessage('Invalid category ID'),
    body('tags')
        .optional()
        .isArray()
        .withMessage('Tags must be an array'),
    body('status')
        .optional()
        .isIn(['draft', 'published', 'archived'])
        .withMessage('Invalid status')
], async (req, res) => {
    try {
        // Check validation errors
        const errors = validationResult(req);
        if (!errors.isEmpty()) {
            return res.status(400).json({
                message: 'Validation failed',
                errors: errors.array()
            });
        }

        const {
            title,
            content,
            excerpt,
            category,
            tags,
            status = 'draft',
            featured = false
        } = req.body;

        // Generate slug from title
        const slug = title
            .toLowerCase()
            .replace(/[^a-z0-9]+/g, '-')
            .replace(/(^-|-$)/g, '');

        // Check if slug already exists
        const existingPost = await Post.findOne({ slug });
        if (existingPost) {
            return res.status(400).json({
                message: 'A post with this title already exists'
            });
        }

        // Create post
        const post = new Post({
            title,
            slug,
            content,
            excerpt: excerpt || content.substring(0, 297) + '...',
            author: req.user.id,
            category,
            tags,
            status,
            featured,
            publishDate: status === 'published' ? new Date() : null
        });

        await post.save();

        // Populate references before sending response
        await post.populate([
            { path: 'author', select: 'name email' },
            { path: 'category', select: 'name slug' }
        ]);

        res.status(201).json({
            message: 'Post created successfully',
            post
        });

    } catch (error) {
        console.error('Error creating post:', error);
        res.status(500).json({ message: 'Server error' });
    }
});

// @route   PUT /api/posts/:id
// @desc    Update post
// @access  Private (Author only)
router.put('/:id', [
    auth,
    body('title')
        .optional()
        .trim()
        .isLength({ min: 5, max: 200 })
        .withMessage('Title must be between 5 and 200 characters'),
    body('content')
        .optional()
        .trim()
        .isLength({ min: 50 })
        .withMessage('Content must be at least 50 characters')
], async (req, res) => {
    try {
        const errors = validationResult(req);
        if (!errors.isEmpty()) {
            return res.status(400).json({
                message: 'Validation failed',
                errors: errors.array()
            });
        }

        const post = await Post.findById(req.params.id);
        
        if (!post) {
            return res.status(404).json({ message: 'Post not found' });
        }

        // Check if user is the author or admin
        if (post.author.toString() !== req.user.id && req.user.role !== 'admin') {
            return res.status(403).json({ message: 'Not authorized to update this post' });
        }

        // Update fields
        const updateFields = ['title', 'content', 'excerpt', 'category', 'tags', 'status', 'featured'];
        updateFields.forEach(field => {
            if (req.body[field] !== undefined) {
                post[field] = req.body[field];
            }
        });

        // Update slug if title changed
        if (req.body.title) {
            const newSlug = req.body.title
                .toLowerCase()
                .replace(/[^a-z0-9]+/g, '-')
                .replace(/(^-|-$)/g, '');
            post.slug = newSlug;
        }

        // Set publish date if status changed to published
        if (req.body.status === 'published' && !post.publishDate) {
            post.publishDate = new Date();
        }

        post.updatedAt = new Date();
        await post.save();

        await post.populate([
            { path: 'author', select: 'name email' },
            { path: 'category', select: 'name slug' }
        ]);

        res.json({
            message: 'Post updated successfully',
            post
        });

    } catch (error) {
        console.error('Error updating post:', error);
        res.status(500).json({ message: 'Server error' });
    }
});

// @route   DELETE /api/posts/:id
// @desc    Delete post
// @access  Private (Author only)
router.delete('/:id', auth, async (req, res) => {
    try {
        const post = await Post.findById(req.params.id);
        
        if (!post) {
            return res.status(404).json({ message: 'Post not found' });
        }

        // Check if user is the author or admin
        if (post.author.toString() !== req.user.id && req.user.role !== 'admin') {
            return res.status(403).json({ message: 'Not authorized to delete this post' });
        }

        await Post.findByIdAndDelete(req.params.id);

        res.json({ message: 'Post deleted successfully' });

    } catch (error) {
        console.error('Error deleting post:', error);
        res.status(500).json({ message: 'Server error' });
    }
});

// @route   POST /api/posts/:id/comments
// @desc    Add comment to post
// @access  Private
router.post('/:id/comments', [
    auth,
    body('content')
        .trim()
        .isLength({ min: 1, max: 500 })
        .withMessage('Comment must be between 1 and 500 characters')
], async (req, res) => {
    try {
        const errors = validationResult(req);
        if (!errors.isEmpty()) {
            return res.status(400).json({
                message: 'Validation failed',
                errors: errors.array()
            });
        }

        const post = await Post.findById(req.params.id);
        
        if (!post) {
            return res.status(404).json({ message: 'Post not found' });
        }

        const comment = {
            content: req.body.content,
            author: req.user.id,
            createdAt: new Date()
        };

        post.comments.push(comment);
        await post.save();

        // Populate the new comment's author info
        await post.populate('comments.author', 'name email');
        
        const newComment = post.comments[post.comments.length - 1];

        res.status(201).json({
            message: 'Comment added successfully',
            comment: newComment
        });

    } catch (error) {
        console.error('Error adding comment:', error);
        res.status(500).json({ message: 'Server error' });
    }
});

// @route   GET /api/posts/stats/overview
// @desc    Get posts statistics
// @access  Private (Admin only)
router.get('/stats/overview', auth, async (req, res) => {
    try {
        if (req.user.role !== 'admin') {
            return res.status(403).json({ message: 'Admin access required' });
        }

        const stats = await Post.aggregate([
            {
                $group: {
                    _id: '$status',
                    count: { $sum: 1 },
                    totalViews: { $sum: '$views' }
                }
            }
        ]);

        const totalPosts = await Post.countDocuments();
        const totalViews = await Post.aggregate([
            { $group: { _id: null, total: { $sum: '$views' } } }
        ]);

        const recentPosts = await Post.find()
            .sort({ createdAt: -1 })
            .limit(5)
            .select('title views createdAt status')
            .populate('author', 'name');

        res.json({
            stats,
            totalPosts,
            totalViews: totalViews[0]?.total || 0,
            recentPosts
        });

    } catch (error) {
        console.error('Error fetching stats:', error);
        res.status(500).json({ message: 'Server error' });
    }
});

module.exports = router;
```

## Dataset 2: Mongoose Models and Middleware
```javascript
// models/User.js
const mongoose = require('mongoose');
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');

const userSchema = new mongoose.Schema({
    name: {
        type: String,
        required: [true, 'Name is required'],
        trim: true,
        maxlength: [50, 'Name cannot exceed 50 characters']
    },
    email: {
        type: String,
        required: [true, 'Email is required'],
        unique: true,
        lowercase: true,
        trim: true,
        match: [
            /^\w+([.-]?\w+)*@\w+([.-]?\w+)*(\.\w{2,3})+$/,
            'Please enter a valid email'
        ]
    },
    password: {
        type: String,
        required: [true, 'Password is required'],
        minlength: [6, 'Password must be at least 6 characters'],
        select: false // Don't include in queries by default
    },
    avatar: {
        type: String,
        default: ''
    },
    bio: {
        type: String,
        maxlength: [500, 'Bio cannot exceed 500 characters'],
        default: ''
    },
    role: {
        type: String,
        enum: ['user', 'admin'],
        default: 'user'
    },
    isActive: {
        type: Boolean,
        default: true
    },
    emailVerified: {
        type: Boolean,
        default: false
    },
    lastLogin: {
        type: Date,
        default: null
    },
    socialMedia: {
        twitter: String,
        github: String,
        linkedin: String,
        website: String
    },
    preferences: {
        emailNotifications: {
            type: Boolean,
            default: true
        },
        theme: {
            type: String,
            enum: ['light', 'dark'],
            default: 'light'
        }
    }
}, {
    timestamps: true,
    toJSON: { virtuals: true },
    toObject: { virtuals: true }
});

// Indexes
userSchema.index({ email: 1 });
userSchema.index({ role: 1, isActive: 1 });

// Virtual for posts count
userSchema.virtual('postsCount', {
    ref: 'Post',
    localField: '_id',
    foreignField: 'author',
    count: true
});

// Virtual for full profile
userSchema.virtual('profile').get(function() {
    return {
        id: this._id,
        name: this.name,
        email: this.email,
        avatar: this.avatar,
        bio: this.bio,
        socialMedia: this.socialMedia,
        joinedAt: this.createdAt
    };
});

// Pre-save middleware to hash password
userSchema.pre('save', async function(next) {
    // Only hash password if it's modified
    if (!this.isModified('password')) {
        return next();
    }

    try {
        // Hash password with cost of 12
        const salt = await bcrypt.genSalt(12);
        this.password = await bcrypt.hash(this.password, salt);
        next();
    } catch (error) {
        next(error);
    }
});

// Method to compare password
userSchema.methods.comparePassword = async function(candidatePassword) {
    return await bcrypt.compare(candidatePassword, this.password);
};

// Method to generate JWT token
userSchema.methods.generateAuthToken = function() {
    return jwt.sign(
        { 
            id: this._id, 
            email: this.email,
            role: this.role 
        },
        process.env.JWT_SECRET,
        { expiresIn: process.env.JWT_EXPIRE || '30d' }
    );
};

// Method to generate password reset token
userSchema.methods.generatePasswordResetToken = function() {
    const resetToken = jwt.sign(
        { id: this._id },
        process.env.JWT_SECRET + this.password,
        { expiresIn: '1h' }
    );
    return resetToken;
};

// Static method to find by credentials
userSchema.statics.findByCredentials = async function(email, password) {
    const user = await this.findOne({ email, isActive: true }).select('+password');
    
    if (!user) {
        throw new Error('Invalid login credentials');
    }

    const isMatch = await user.comparePassword(password);
    if (!isMatch) {
        throw new Error('Invalid login credentials');
    }

    return user;
};

module.exports = mongoose.model('User', userSchema);

// models/Post.js
const mongoose = require('mongoose');
const slugify = require('slugify');

const commentSchema = new mongoose.Schema({
    content: {
        type: String,
        required: true,
        trim: true,
        maxlength: [500, 'Comment cannot exceed 500 characters']
    },
    author: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'User',
        required: true
    },
    approved: {
        type: Boolean,
        default: true
    },
    likes: [{
        type: mongoose.Schema.Types.ObjectId,
        ref: 'User'
    }],
    replies: [{
        content: String,
        author: {
            type: mongoose.Schema.Types.ObjectId,
            ref: 'User'
        },
        createdAt: {
            type: Date,
            default: Date.now
        }
    }]
}, {
    timestamps: true
});

const postSchema = new mongoose.Schema({
    title: {
        type: String,
        required: [true, 'Title is required'],
        trim: true,
        maxlength: [200, 'Title cannot exceed 200 characters']
    },
    slug: {
        type: String,
        unique: true,
        lowercase: true,
        index: true
    },
    excerpt: {
        type: String,
        maxlength: [300, 'Excerpt cannot exceed 300 characters']
    },
    content: {
        type: String,
        required: [true, 'Content is required'],
        minlength: [50, 'Content must be at least 50 characters']
    },
    author: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'User',
        required: true
    },
    category: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'Category'
    },
    tags: [{
        type: String,
        trim: true,
        lowercase: true
    }],
    status: {
        type: String,
        enum: ['draft', 'published', 'archived'],
        default: 'draft'
    },
    featured: {
        type: Boolean,
        default: false
    },
    publishDate: {
        type: Date,
        default: null
    },
    views: {
        type: Number,
        default: 0
    },
    likes: [{
        user: {
            type: mongoose.Schema.Types.ObjectId,
            ref: 'User'
        },
        createdAt: {
            type: Date,
            default: Date.now
        }
    }],
    comments: [commentSchema],
    readingTime: {
        type: Number, // in minutes
        default: 0
    },
    seoMeta: {
        metaTitle: String,
        metaDescription: String,
        keywords: [String]
    }
}, {
    timestamps: true,
    toJSON: { virtuals: true },
    toObject: { virtuals: true }
});

// Indexes for better performance
postSchema.index({ status: 1, publishDate: -1 });
postSchema.index({ author: 1, status: 1 });
postSchema.index({ slug: 1 });
postSchema.index({ tags: 1 });
postSchema.index({ '$**': 'text' }); // Text index for search

// Virtual for comment count
postSchema.virtual('commentCount').get(function() {
    return this.comments ? this.comments.length : 0;
});

// Virtual for like count
postSchema.virtual('likeCount').get(function() {
    return this.likes ? this.likes.length : 0;
});

// Virtual for reading time calculation
postSchema.virtual('estimatedReadingTime').get(function() {
    if (!this.content) return 0;
    const wordsPerMinute = 200;
    const wordCount = this.content.split(' ').length;
    return Math.ceil(wordCount / wordsPerMinute);
});

// Pre-save middleware
postSchema.pre('save', function(next) {
    // Generate slug if not provided
    if (this.isModified('title') && !this.slug) {
        this.slug = slugify(this.title, {
            lower: true,
            strict: true,
            remove: /[*+~.()'"!:@]/g
        });
    }

    // Generate excerpt if not provided
    if (this.isModified('content') && !this.excerpt) {
        this.excerpt = this.content.length > 297 
            ? this.content.substring(0, 297) + '...'
            : this.content;
    }

    // Calculate reading time
    if (this.isModified('content')) {
        this.readingTime = this.estimatedReadingTime;
    }

    // Set publish date if status changed to published
    if (this.isModified('status') && this.status === 'published' && !this.publishDate) {
        this.publishDate = new Date();
    }

    next();
});

// Static methods
postSchema.statics.findPublished = function() {
    return this.find({ 
        status: 'published', 
        publishDate: { $lte: new Date() } 
    });
};

postSchema.statics.findByTag = function(tag) {
    return this.find({ 
        tags: { $in: [tag] },
        status: 'published' 
    });
};

postSchema.statics.getPopular = function(limit = 10) {
    return this.find({ status: 'published' })
        .sort({ views: -1, likes: -1 })
        .limit(limit);
};

// Instance methods
postSchema.methods.incrementViews = function() {
    this.views += 1;
    return this.save();
};

postSchema.methods.addLike = function(userId) {
    if (!this.likes.some(like => like.user.toString() === userId.toString())) {
        this.likes.push({ user: userId });
        return this.save();
    }
    return this;
};

postSchema.methods.removeLike = function(userId) {
    this.likes = this.likes.filter(like => like.user.toString() !== userId.toString());
    return this.save();
};

postSchema.methods.addComment = function(commentData) {
    this.comments.push(commentData);
    return this.save();
};

postSchema.methods.getRelatedPosts = async function(limit = 5) {
    const Post = this.constructor;
    
    // Find posts with similar tags or same category
    const related = await Post.find({
        _id: { $ne: this._id },
        status: 'published',
        $or: [
            { tags: { $in: this.tags } },
            { category: this.category }
        ]
    })
    .populate('author', 'name')
    .sort({ publishDate: -1 })
    .limit(limit);
    
    return related;
};

module.exports = mongoose.model('Post', postSchema);

// models/Category.js
const mongoose = require('mongoose');

const categorySchema = new mongoose.Schema({
    name: {
        type: String,
        required: [true, 'Category name is required'],
        unique: true,
        trim: true,
        maxlength: [50, 'Category name cannot exceed 50 characters']
    },
    slug: {
        type: String,
        unique: true,
        lowercase: true
    },
    description: {
        type: String,
        maxlength: [200, 'Description cannot exceed 200 characters']
    },
    color: {
        type: String,
        match: [/^#[0-9A-F]{6}$/i, 'Please enter a valid hex color']
    },
    icon: String,
    parent: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'Category',
        default: null
    },
    isActive: {
        type: Boolean,
        default: true
    }
}, {
    timestamps: true,
    toJSON: { virtuals: true },
    toObject: { virtuals: true }
});

// Virtual for posts count
categorySchema.virtual('postsCount', {
    ref: 'Post',
    localField: '_id',
    foreignField: 'category',
    count: true
});

// Virtual for subcategories
categorySchema.virtual('subcategories', {
    ref: 'Category',
    localField: '_id',
    foreignField: 'parent'
});

// Pre-save middleware to generate slug
categorySchema.pre('save', function(next) {
    if (this.isModified('name')) {
        this.slug = this.name
            .toLowerCase()
            .replace(/[^a-z0-9]+/g, '-')
            .replace(/(^-|-$)/g, '');
    }
    next();
});

// Static method to get category tree
categorySchema.statics.getCategoryTree = async function() {
    const categories = await this.find({ parent: null })
        .populate('subcategories')
        .sort({ name: 1 });
    return categories;
};

module.exports = mongoose.model('Category', categorySchema);
```

## Dataset 3: Middleware and Authentication
```javascript
// middleware/auth.js - JWT Authentication middleware
const jwt = require('jsonwebtoken');
const User = require('../models/User');

const auth = async (req, res, next) => {
    try {
        // Get token from header
        const authHeader = req.header('Authorization');
        
        if (!authHeader || !authHeader.startsWith('Bearer ')) {
            return res.status(401).json({ 
                message: 'No token provided, authorization denied' 
            });
        }

        // Extract token
        const token = authHeader.replace('Bearer ', '');

        // Verify token
        const decoded = jwt.verify(token, process.env.JWT_SECRET);

        // Get user from token
        const user = await User.findById(decoded.id).select('-password');
        
        if (!user) {
            return res.status(401).json({ 
                message: 'Token is not valid' 
            });
        }

        if (!user.isActive) {
            return res.status(401).json({ 
                message: 'Account is deactivated' 
            });
        }

        // Add user to request object
        req.user = user;
        next();

    } catch (error) {
        console.error('Auth middleware error:', error);
        
        if (error.name === 'JsonWebTokenError') {
            return res.status(401).json({ message: 'Token is not valid' });
        }
        
        if (error.name === 'TokenExpiredError') {
            return res.status(401).json({ message: 'Token has expired' });
        }

        res.status(500).json({ message: 'Server error in authentication' });
    }
};

// Role-based authorization middleware
const authorize = (...roles) => {
    return (req, res, next) => {
        if (!req.user) {
            return res.status(401).json({ 
                message: 'Authentication required' 
            });
        }

        if (!roles.includes(req.user.role)) {
            return res.status(403).json({ 
                message: `Access denied. Required roles: ${roles.join(', ')}` 
            });
        }

        next();
    };
};

// Optional auth middleware (doesn't fail if no token)
const optionalAuth = async (req, res, next) => {
    try {
        const authHeader = req.header('Authorization');
        
        if (authHeader && authHeader.startsWith('Bearer ')) {
            const token = authHeader.replace('Bearer ', '');
            const decoded = jwt.verify(token, process.env.JWT_SECRET);
            const user = await User.findById(decoded.id).select('-password');
            
            if (user && user.isActive) {
                req.user = user;
            }
        }
        
        next();
    } catch (error) {
        // Continue without authentication
        next();
    }
};

module.exports = { auth, authorize, optionalAuth };

// middleware/validation.js - Request validation middleware
const { body, query, param, validationResult } = require('express-validator');

// Validation error handler
const handleValidationErrors = (req, res, next) => {
    const errors = validationResult(req);
    
    if (!errors.isEmpty()) {
        const errorMessages = errors.array().map(error => ({
            field: error.param,
            message: error.msg,
            value: error.value
        }));

        return res.status(400).json({
            message: 'Validation failed',
            errors: errorMessages
        });
    }
    
    next();
};

// User validation rules
const validateUserRegistration = [
    body('name')
        .trim()
        .isLength({ min: 2, max: 50 })
        .withMessage('Name must be between 2 and 50 characters')
        .matches(/^[a-zA-Z\s]+$/)
        .withMessage('Name can only contain letters and spaces'),
        
    body('email')
        .isEmail()
        .normalizeEmail()
        .withMessage('Please provide a valid email'),
        
    body('password')
        .isLength({ min: 6, max: 128 })
        .withMessage('Password must be between 6 and 128 characters')
        .matches(/^(?=.*[a-z])(?=.*[A-Z])(?=.*\d)/)
        .withMessage('Password must contain at least one lowercase letter, one uppercase letter, and one number'),
        
    body('confirmPassword')
        .custom((value, { req }) => {
            if (value !== req.body.password) {
                throw new Error('Password confirmation does not match password');
            }
            return true;
        }),
        
    handleValidationErrors
];

const validateUserLogin = [
    body('email')
        .isEmail()
        .normalizeEmail()
        .withMessage('Please provide a valid email'),
        
    body('password')
        .notEmpty()
        .withMessage('Password is required'),
        
    handleValidationErrors
];

// Post validation rules
const validatePost = [
    body('title')
        .trim()
        .isLength({ min: 5, max: 200 })
        .withMessage('Title must be between 5 and 200 characters'),
        
    body('content')
        .trim()
        .isLength({ min: 50 })
        .withMessage('Content must be at least 50 characters'),
        
    body('excerpt')
        .optional()
        .trim()
        .isLength({ max: 300 })
        .withMessage('Excerpt cannot exceed 300 characters'),
        
    body('category')
        .optional()
        .isMongoId()
        .withMessage('Invalid category ID'),
        
    body('tags')
        .optional()
        .isArray()
        .withMessage('Tags must be an array')
        .custom((tags) => {
            if (tags.length > 10) {
                throw new Error('Cannot have more than 10 tags');
            }
            return true;
        }),
        
    body('status')
        .optional()
        .isIn(['draft', 'published', 'archived'])
        .withMessage('Status must be draft, published, or archived'),
        
    handleValidationErrors
];

// Query validation rules
const validatePagination = [
    query('page')
        .optional()
        .isInt({ min: 1 })
        .withMessage('Page must be a positive integer'),
        
    query('limit')
        .optional()
        .isInt({ min: 1, max: 100 })
        .withMessage('Limit must be between 1 and 100'),
        
    handleValidationErrors
];

const validateSearch = [
    query('search')
        .optional()
        .trim()
        .isLength({ min: 1, max: 100 })
        .withMessage('Search query must be between 1 and 100 characters'),
        
    handleValidationErrors
];

// Parameter validation
const validateObjectId = [
    param('id')
        .isMongoId()
        .withMessage('Invalid ID format'),
        
    handleValidationErrors
];

module.exports = {
    validateUserRegistration,
    validateUserLogin,
    validatePost,
    validatePagination,
    validateSearch,
    validateObjectId,
    handleValidationErrors
};

// middleware/errorHandler.js - Global error handling
const errorHandler = (err, req, res, next) => {
    let error = { ...err };
    error.message = err.message;

    // Log error
    console.error('Error:', err);

    // Mongoose bad ObjectId
    if (err.name === 'CastError') {
        const message = 'Resource not found';
        error = { message, status: 404 };
    }

    // Mongoose duplicate key
    if (err.code === 11000) {
        const field = Object.keys(err.keyValue)[0];
        const message = `${field} already exists`;
        error = { message, status: 400 };
    }

    // Mongoose validation error
    if (err.name === 'ValidationError') {
        const message = Object.values(err.errors).map(val => val.message);
        error = { message, status: 400 };
    }

    // JWT errors
    if (err.name === 'JsonWebTokenError') {
        const message = 'Invalid token';
        error = { message, status: 401 };
    }

    if (err.name === 'TokenExpiredError') {
        const message = 'Token expired';
        error = { message, status: 401 };
    }

    // File upload errors
    if (err.code === 'LIMIT_FILE_SIZE') {
        const message = 'File size too large';
        error = { message, status: 400 };
    }

    res.status(error.status || 500).json({
        success: false,
        message: error.message || 'Server Error',
        ...(process.env.NODE_ENV === 'development' && { stack: err.stack })
    });
};

// middleware/notFound.js - 404 handler
const notFound = (req, res, next) => {
    const error = new Error(`Route ${req.originalUrl} not found`);
    res.status(404).json({
        success: false,
        message: error.message
    });
};

// middleware/logger.js - Custom logging middleware
const logger = (req, res, next) => {
    const start = Date.now();
    
    // Override res.end to log when response is finished
    const originalEnd = res.end;
    res.end = function(...args) {
        const duration = Date.now() - start;
        
        console.log(`${req.method} ${req.originalUrl} - ${res.statusCode} - ${duration}ms`);
        
        // Log additional info for errors
        if (res.statusCode >= 400) {
            console.log(`Error details: ${req.method} ${req.originalUrl}`, {
                ip: req.ip,
                userAgent: req.get('User-Agent'),
                body: req.body,
                query: req.query,
                params: req.params
            });
        }
        
        originalEnd.apply(this, args);
    };
    
    next();
};

// middleware/security.js - Additional security measures
const rateLimit = require('express-rate-limit');

// Different rate limits for different routes
const authLimiter = rateLimit({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 5, // 5 attempts per window
    message: {
        error: 'Too many authentication attempts, please try again later'
    },
    standardHeaders: true,
    legacyHeaders: false
});

const apiLimiter = rateLimit({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 100, // 100 requests per window
    message: {
        error: 'Too many API requests, please try again later'
    }
});

// XSS protection
const xssProtection = (req, res, next) => {
    // Basic XSS protection for JSON bodies
    if (req.body) {
        for (const key in req.body) {
            if (typeof req.body[key] === 'string') {
                req.body[key] = req.body[key]
                    .replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '')
                    .replace(/javascript:/gi, '')
                    .replace(/on\w+\s*=/gi, '');
            }
        }
    }
    next();
};

module.exports = {
    errorHandler,
    notFound,
    logger,
    authLimiter,
    apiLimiter,
    xssProtection
};
```