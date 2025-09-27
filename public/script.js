// Modern JavaScript for Nibert Investments DATA website
// ES6+ features with professional animations and download functionality

class DatasetDownloader {
    constructor() {
        this.loadingOverlay = document.getElementById('loadingOverlay');
        this.downloadButtons = document.querySelectorAll('.download-btn');
        this.init();
    }

    init() {
        this.bindEvents();
        this.initAnimations();
        this.initScrollEffects();
        this.initContactForm();
    }

    bindEvents() {
        // Download button events
        this.downloadButtons.forEach(button => {
            button.addEventListener('click', (e) => this.handleDownload(e));
        });

        // Smooth scrolling for navigation links (only for hash links)
        document.querySelectorAll('.nav-link').forEach(link => {
            link.addEventListener('click', (e) => this.handleNavClick(e));
        });

        // Header scroll effect
        window.addEventListener('scroll', () => this.handleScroll());

        // Add loading states to interactive elements
        this.addLoadingStates();
    }

    initContactForm() {
        const contactForm = document.getElementById('contactForm');
        const messageTextarea = document.getElementById('message');
        const charCount = document.getElementById('charCount');
        
        if (contactForm && messageTextarea && charCount) {
            // Character counting
            messageTextarea.addEventListener('input', () => {
                const length = messageTextarea.value.length;
                charCount.textContent = length;
                
                if (length > 700) {
                    charCount.style.color = '#dc2626';
                } else if (length > 600) {
                    charCount.style.color = '#f59e0b';
                } else {
                    charCount.style.color = '#6b7280';
                }
            });

            // Form submission
            contactForm.addEventListener('submit', (e) => this.handleContactSubmit(e));

            // Input focus effects
            const inputs = contactForm.querySelectorAll('input, textarea');
            inputs.forEach(input => {
                input.addEventListener('focus', () => {
                    input.style.borderColor = '#dc2626';
                    input.style.boxShadow = '0 0 0 3px rgba(220, 38, 38, 0.1)';
                });
                
                input.addEventListener('blur', () => {
                    input.style.borderColor = '#e2e8f0';
                    input.style.boxShadow = 'none';
                });
            });
        }
    }

    handleContactSubmit(event) {
        event.preventDefault();
        
        const form = event.target;
        const formData = new FormData(form);
        const subject = formData.get('subject');
        const message = formData.get('message');
        
        if (!subject || !message) {
            this.showError('Please fill in all required fields.');
            return;
        }

        if (message.length > 750) {
            this.showError('Message must be 750 characters or less.');
            return;
        }

        // Create mailto link
        const email = 'josh@nibertinvestments.com';
        const encodedSubject = encodeURIComponent(`[DATA Website] ${subject}`);
        const encodedMessage = encodeURIComponent(message);
        const mailtoLink = `mailto:${email}?subject=${encodedSubject}&body=${encodedMessage}`;
        
        // Open email client
        window.location.href = mailtoLink;
        
        // Show success message
        this.showSuccess('Email client opened! Please send the email from your email application.');
        
        // Reset form
        form.reset();
        document.getElementById('charCount').textContent = '0';
    }

    handleNavClick(event) {
        const href = event.target.getAttribute('href');
        
        // Only handle smooth scrolling for hash links (anchor links on same page)
        if (href && href.startsWith('#')) {
            event.preventDefault();
            const target = document.querySelector(href);
            if (target) {
                const headerHeight = 80;
                const targetPosition = target.offsetTop - headerHeight;
                
                window.scrollTo({
                    top: targetPosition,
                    behavior: 'smooth'
                });
            }
        }
        // For regular page links, let the browser handle normal navigation
    }

    handleScroll() {
        const header = document.querySelector('.header');
        const scrolled = window.pageYOffset > 50;
        
        if (scrolled) {
            header.style.background = 'rgba(255, 255, 255, 0.98)';
            header.style.boxShadow = '0 2px 20px rgba(0, 0, 0, 0.1)';
        } else {
            header.style.background = 'rgba(255, 255, 255, 0.95)';
            header.style.boxShadow = 'none';
        }
    }

    initAnimations() {
        // Animate elements on scroll
        this.observeElements();
        
        // Add hover effects to cards
        this.addCardEffects();
        
        // Animate code preview typing effect
        this.animateCodePreview();
    }

    observeElements() {
        const observerOptions = {
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        };

        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.style.opacity = '1';
                    entry.target.style.transform = 'translateY(0)';
                }
            });
        }, observerOptions);

        // Observe cards and sections
        document.querySelectorAll('.feature-card, .dataset-card, .section-title').forEach(el => {
            el.style.opacity = '0';
            el.style.transform = 'translateY(30px)';
            el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
            observer.observe(el);
        });
    }

    addCardEffects() {
        document.querySelectorAll('.feature-card, .dataset-card').forEach(card => {
            card.addEventListener('mouseenter', () => {
                card.style.transform = 'translateY(-8px) scale(1.02)';
            });
            
            card.addEventListener('mouseleave', () => {
                card.style.transform = 'translateY(0) scale(1)';
            });
        });
    }

    animateCodePreview() {
        const codeLines = document.querySelectorAll('.code-line');
        
        codeLines.forEach((line, index) => {
            line.style.opacity = '0';
            line.style.transform = 'translateX(-20px)';
            
            setTimeout(() => {
                line.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
                line.style.opacity = '1';
                line.style.transform = 'translateX(0)';
            }, index * 200);
        });
    }

    initScrollEffects() {
        // Parallax effect for hero section
        window.addEventListener('scroll', () => {
            const scrolled = window.pageYOffset;
            const heroVisual = document.querySelector('.hero-visual');
            
            if (heroVisual) {
                const rate = scrolled * -0.3;
                heroVisual.style.transform = `translateY(${rate}px)`;
            }
        });

        // Progress indicator
        this.createProgressIndicator();
    }

    createProgressIndicator() {
        const progressBar = document.createElement('div');
        progressBar.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 0%;
            height: 3px;
            background: linear-gradient(90deg, #dc2626, #991b1b);
            z-index: 9999;
            transition: width 0.3s ease;
        `;
        document.body.appendChild(progressBar);

        window.addEventListener('scroll', () => {
            const scrollPercent = (window.pageYOffset / (document.body.scrollHeight - window.innerHeight)) * 100;
            progressBar.style.width = `${Math.min(scrollPercent, 100)}%`;
        });
    }

    addLoadingStates() {
        document.querySelectorAll('button, .btn').forEach(button => {
            button.addEventListener('click', function() {
                if (!this.classList.contains('loading')) {
                    this.classList.add('loading');
                    setTimeout(() => {
                        this.classList.remove('loading');
                    }, 2000);
                }
            });
        });
    }

    async handleDownload(event) {
        event.preventDefault();
        
        try {
            this.showLoading();
            
            // Simulate preparation time for better UX
            await this.delay(1500);
            
            // Create and download the ZIP file
            await this.createDatasetZip();
            
        } catch (error) {
            console.error('Download failed:', error);
            this.showError('Download failed. Please try again.');
        } finally {
            this.hideLoading();
        }
    }

    async createDatasetZip() {
        try {
            // Primary method: Use GitHub API to get complete repository ZIP
            const response = await fetch('https://api.github.com/repos/nibertinvestments/DATA/zipball/main');
            
            if (response.ok) {
                const blob = await response.blob();
                this.downloadBlob(blob, 'nibert-investments-data-complete.zip');
                this.showSuccess('Complete repository downloaded successfully!');
                return;
            }
            
            // If API fails, use direct GitHub download link
            console.log('GitHub API failed, using direct download link');
            this.downloadDirectFromGitHub();
            
        } catch (error) {
            console.error('GitHub API failed, using direct download link:', error);
            this.downloadDirectFromGitHub();
        }
    }

    downloadDirectFromGitHub() {
        // Create info file to explain the download
        this.createFallbackZip();
        
        // Open direct GitHub download link in new tab
        setTimeout(() => {
            window.open('https://github.com/nibertinvestments/DATA/archive/refs/heads/main.zip', '_blank');
            this.showSuccess('Download started from GitHub. Check your downloads folder.');
        }, 500);
    }

    createFallbackZip() {
        // Create a comprehensive info file about the datasets
        const datasetInfo = `
# Nibert Investments DATA Repository
# Complete ML Datasets and AI Training Data

## Repository Contents
- 106+ curated code examples across multiple programming languages
- Repository size: ~6.9MB
- 46 documentation files
- Organized testing structure for validation

## Directory Structure
- data-sources/: Programming language examples and patterns
- code_samples/: Curated code samples for AI training (106 files)
- high_end_specialized/: Advanced algorithms and financial mathematics
- scripts/: Data processing and ML pipeline scripts
- documentation/: Specifications and tutorials
- tests/: Validation and testing frameworks

## Access Methods
1. Direct GitHub Download: https://github.com/nibertinvestments/DATA/archive/refs/heads/main.zip
2. Git Clone: git clone https://github.com/nibertinvestments/DATA.git
3. GitHub API: https://api.github.com/repos/nibertinvestments/DATA/zipball/main

## Contact Information
- Company: Nibert Investments LLC
- Email: josh@nibertinvestements.com
- Repository: https://github.com/nibertinvestments/DATA

## Dataset Categories
### Languages (20 supported)
- Python: 16 files - ML algorithms, data structures, core patterns
- Kotlin: 9 files - JVM/Android development, modern patterns
- Java: 8 files - Enterprise patterns, neural networks, robust design
- Go: 6 files - Concurrent programming, systems development
- JavaScript: 6 files - Modern ES6+, async programming, web patterns
- Rust: 6 files - Memory-safe systems programming
- TypeScript: 5 files - Type-safe development, enterprise patterns
- C#: 5 files - .NET enterprise patterns
- C++: 5 files - High-performance computing
- Ruby: 5 files - Web development, metaprogramming
- PHP: 5 files - Modern web frameworks
- Swift: 5 files - iOS/macOS development
- Scala: 5 files - Functional programming, JVM integration
- Dart: 5 files - Flutter, cross-platform development
- C++: 4 files - Performance-critical algorithms, memory management
- Lua: 3 files - Embedded scripting, game development
- Solidity: 3 files - Smart contracts, DeFi protocols, DAO governance
- Perl: 2 files - Text processing, legacy systems
- Elixir: 1 file - Actor model, fault tolerance
- Haskell: 1 file - Pure functional programming
- C++: Performance-critical algorithms, memory management
- Rust: Systems programming, safety-first design
- TypeScript: Type-safe JavaScript development
- Solidity: Smart contracts, DeFi protocols, DAO governance
- Dart: Flutter applications, mobile development
- R: Data science, statistical analysis, machine learning
- Haskell: Functional programming, category theory
- Elixir: Concurrent programming, actor model, OTP
- Lua: Game development, scripting, advanced programming
- Perl: Text processing, system administration, advanced regex
- And more...

### Cross-Language Patterns (25+)
- Algorithm implementations across languages
- Design patterns and best practices
- Universal programming concepts
- Performance optimization techniques
- Intermediate-level examples for all languages

### Frameworks (8+)
- React: Modern UI development
- Django: Python web framework
- Spring: Java enterprise framework
- Express.js: Node.js web framework
- Phoenix: Elixir web framework
- Flutter: Dart mobile framework
- And more...

### Specialized Domains (8+)
- Smart Contracts: Token creation, DAO governance, DEX/AMM
- Data Science: Statistical analysis, ML pipelines, visualization
- Functional Programming: Monads, category theory, immutable structures
- Concurrent Programming: Actor model, STM, async patterns
- Text Processing: Advanced regex, parsing, natural language
- Game Development: Entity systems, scripting, performance
- Cryptography: Advanced encryption, hashing, security
- Blockchain: DeFi protocols, consensus algorithms

Generated on: ${new Date().toISOString()}
Core Code Examples: 106+ curated implementations
Repository Size: ~6.9MB (source), ~1.2MB (ZIP download)
Documentation Files: 46
Test Infrastructure: Organized for validation

## DOWNLOAD INSTRUCTIONS
The complete repository with 106+ curated code examples will be downloaded automatically.
This includes all code samples, documentation, testing structure, and datasets.
        `.trim();

        const blob = new Blob([datasetInfo], { type: 'text/plain' });
        this.downloadBlob(blob, 'nibert-investments-data-info.txt');
    }

    downloadBlob(blob, filename) {
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        // Show success message
        this.showSuccess(`Download started: ${filename}`);
    }

    showLoading() {
        if (this.loadingOverlay) {
            this.loadingOverlay.classList.add('active');
            document.body.style.overflow = 'hidden';
        }
    }

    hideLoading() {
        if (this.loadingOverlay) {
            this.loadingOverlay.classList.remove('active');
            document.body.style.overflow = '';
        }
    }

    showSuccess(message) {
        this.showNotification(message, 'success');
    }

    showError(message) {
        this.showNotification(message, 'error');
    }

    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.style.cssText = `
            position: fixed;
            top: 100px;
            right: 20px;
            padding: 16px 24px;
            border-radius: 8px;
            color: white;
            font-family: var(--font-primary);
            font-weight: 500;
            z-index: 10000;
            transform: translateX(100%);
            transition: transform 0.3s ease;
            max-width: 400px;
            word-wrap: break-word;
        `;

        // Set background based on type
        const colors = {
            success: 'linear-gradient(135deg, #10b981, #059669)',
            error: 'linear-gradient(135deg, #ef4444, #dc2626)',
            info: 'linear-gradient(135deg, #3b82f6, #2563eb)'
        };
        
        notification.style.background = colors[type] || colors.info;
        notification.textContent = message;
        
        document.body.appendChild(notification);
        
        // Animate in
        setTimeout(() => {
            notification.style.transform = 'translateX(0)';
        }, 100);
        
        // Auto remove
        setTimeout(() => {
            notification.style.transform = 'translateX(100%)';
            setTimeout(() => {
                document.body.removeChild(notification);
            }, 300);
        }, 4000);
    }

    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

// Performance optimizations
class PerformanceOptimizer {
    constructor() {
        this.init();
    }

    init() {
        this.optimizeImages();
        this.addPreloadHints();
        this.optimizeAnimations();
    }

    optimizeImages() {
        // Add lazy loading to future images
        document.querySelectorAll('img').forEach(img => {
            img.loading = 'lazy';
        });
    }

    addPreloadHints() {
        // Preload critical resources
        const preloadLinks = [
            { href: 'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap', as: 'style' }
        ];

        preloadLinks.forEach(link => {
            const preloadLink = document.createElement('link');
            preloadLink.rel = 'preload';
            preloadLink.href = link.href;
            preloadLink.as = link.as;
            document.head.appendChild(preloadLink);
        });
    }

    optimizeAnimations() {
        // Reduce motion for users who prefer it
        const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)');
        
        if (prefersReducedMotion.matches) {
            document.documentElement.style.setProperty('--transition-fast', '0ms');
            document.documentElement.style.setProperty('--transition-normal', '0ms');
            document.documentElement.style.setProperty('--transition-slow', '0ms');
        }
    }
}

// Analytics and tracking (privacy-friendly)
class Analytics {
    constructor() {
        this.events = [];
        this.init();
    }

    init() {
        this.trackPageView();
        this.trackInteractions();
    }

    trackPageView() {
        this.logEvent('page_view', {
            url: window.location.href,
            title: document.title,
            timestamp: new Date().toISOString()
        });
    }

    trackInteractions() {
        // Track download attempts
        document.querySelectorAll('.download-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const isPremium = btn.id === 'downloadBtnPremium';
                const isHighEnd = btn.id === 'downloadBtnHighEnd';
                
                let downloadType = 'Complete Library';
                if (isPremium) downloadType = 'Premium Collection';
                if (isHighEnd) downloadType = 'High-End Specialized';
                
                this.logEvent('download_attempt', {
                    button_id: btn.id || 'unknown',
                    download_type: downloadType,
                    timestamp: new Date().toISOString()
                });
                
                // Handle different download types
                if (isPremium) {
                    // For now, redirect to complete download since premium isn't set up yet
                    setTimeout(() => {
                        window.open('https://github.com/nibertinvestments/DATA/archive/refs/heads/main.zip', '_blank');
                    }, 1000);
                } else if (isHighEnd) {
                    // Handle high-end specialized download
                    e.preventDefault();
                    this.handleHighEndDownload();
                }
            });
        });

        // Track navigation
        document.querySelectorAll('.nav-link').forEach(link => {
            link.addEventListener('click', () => {
                this.logEvent('navigation_click', {
                    target: link.getAttribute('href'),
                    timestamp: new Date().toISOString()
                });
            });
        });
    }

    handleHighEndDownload() {
        // Show loading and then download the high-end specialized package
        this.showLoading();
        
        setTimeout(() => {
            // Create the download link for our high-end package
            const link = document.createElement('a');
            link.href = '/high_end_specialized_premium.zip';
            link.download = 'high_end_specialized_premium.zip';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            
            this.hideLoading();
            this.showSuccess('High-End Specialized package downloaded! Contains 15+ algorithms, functions, and equations.');
        }, 1500);
    }

    showLoading() {
        const loadingOverlay = document.getElementById('loadingOverlay');
        if (loadingOverlay) {
            loadingOverlay.classList.add('active');
            document.body.style.overflow = 'hidden';
        }
    }

    hideLoading() {
        const loadingOverlay = document.getElementById('loadingOverlay');
        if (loadingOverlay) {
            loadingOverlay.classList.remove('active');
            document.body.style.overflow = '';
        }
    }

    showSuccess(message) {
        this.showNotification(message, 'success');
    }

    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.style.cssText = `
            position: fixed;
            top: 100px;
            right: 20px;
            padding: 16px 24px;
            border-radius: 8px;
            color: white;
            font-family: var(--font-primary);
            font-weight: 500;
            z-index: 10000;
            transform: translateX(100%);
            transition: transform 0.3s ease;
            max-width: 400px;
            word-wrap: break-word;
        `;

        // Set background based on type
        const colors = {
            success: 'linear-gradient(135deg, #10b981, #059669)',
            error: 'linear-gradient(135deg, #ef4444, #dc2626)',
            info: 'linear-gradient(135deg, #3b82f6, #2563eb)'
        };
        
        notification.style.background = colors[type] || colors.info;
        notification.textContent = message;
        
        document.body.appendChild(notification);
        
        // Animate in
        setTimeout(() => {
            notification.style.transform = 'translateX(0)';
        }, 100);
        
        // Auto remove
        setTimeout(() => {
            notification.style.transform = 'translateX(100%)';
            setTimeout(() => {
                if (document.body.contains(notification)) {
                    document.body.removeChild(notification);
                }
            }, 300);
        }, 4000);
    }

    logEvent(eventName, eventData) {
        this.events.push({ name: eventName, data: eventData });
        
        // In a real implementation, you might send this to an analytics service
        console.log(`Analytics Event: ${eventName}`, eventData);
    }
}

// Initialize everything when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new DatasetDownloader();
    new PerformanceOptimizer();
    new Analytics();
    
    // Add some final touches
    console.log('🚀 Nibert Investments DATA website loaded successfully!');
    console.log('📊 Professional ML Datasets for AI Coding Agents');
    console.log('🔗 GitHub: https://github.com/nibertinvestments/DATA');
    
    // Easter egg for developers
    console.log(`
    ███╗   ██╗██╗██████╗ ███████╗██████╗ ████████╗
    ████╗  ██║██║██╔══██╗██╔════╝██╔══██╗╚══██╔══╝
    ██╔██╗ ██║██║██████╔╝█████╗  ██████╔╝   ██║   
    ██║╚██╗██║██║██╔══██╗██╔══╝  ██╔══██╗   ██║   
    ██║ ╚████║██║██████╔╝███████╗██║  ██║   ██║   
    ╚═╝  ╚═══╝╚═╝╚═════╝ ╚══════╝╚═╝  ╚═╝   ╚═╝   
    
    Thanks for checking out our code! 
    We're hiring talented developers.
    Email: josh@nibertinvestements.com
    `);
});