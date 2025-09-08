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
    }

    bindEvents() {
        // Download button events
        this.downloadButtons.forEach(button => {
            button.addEventListener('click', (e) => this.handleDownload(e));
        });

        // Smooth scrolling for navigation links
        document.querySelectorAll('.nav-link').forEach(link => {
            link.addEventListener('click', (e) => this.handleNavClick(e));
        });

        // Header scroll effect
        window.addEventListener('scroll', () => this.handleScroll());

        // Add loading states to interactive elements
        this.addLoadingStates();
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
            // Use modern fetch API to gather all repository content
            const response = await fetch('https://api.github.com/repos/nibertinvestments/DATA/zipball/main');
            
            if (!response.ok) {
                // Fallback: Create a simulated ZIP with available data
                this.createFallbackZip();
                return;
            }

            const blob = await response.blob();
            this.downloadBlob(blob, 'nibert-investments-data-complete.zip');
            
        } catch (error) {
            console.error('GitHub API failed, using fallback:', error);
            this.createFallbackZip();
        }
    }

    createFallbackZip() {
        // Create a comprehensive info file about the datasets
        const datasetInfo = `
# Nibert Investments DATA Repository
# Complete ML Datasets and AI Training Data

## Repository Contents
- 41+ code files across multiple programming languages
- Total size: ~2.4MB
- Languages: Python, JavaScript, Java, C++, Go, Rust, TypeScript

## Directory Structure
- data-sources/: Programming language examples and patterns
- code_samples/: Comprehensive code samples for AI training
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
### Languages (8+ supported)
- Python: ML algorithms, data structures, best practices
- JavaScript: Modern ES6+, async programming, frameworks
- Java: Enterprise patterns, object-oriented design
- C++: Performance-critical algorithms, memory management
- Go: Concurrent programming, microservices
- Rust: Systems programming, safety-first design
- TypeScript: Type-safe JavaScript development
- And more...

### Cross-Language Patterns (10+)
- Algorithm implementations across languages
- Design patterns and best practices
- Universal programming concepts
- Performance optimization techniques

### Frameworks (15+)
- React: Modern UI development
- Django: Python web framework
- Spring: Java enterprise framework
- Express.js: Node.js web framework
- And many more...

Generated on: ${new Date().toISOString()}
Total Files: 41+
Repository Size: ~2.4MB
        `.trim();

        const blob = new Blob([datasetInfo], { type: 'text/plain' });
        this.downloadBlob(blob, 'nibert-investments-data-info.txt');
        
        // Also trigger GitHub download link
        setTimeout(() => {
            window.open('https://github.com/nibertinvestments/DATA/archive/refs/heads/main.zip', '_blank');
        }, 1000);
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

    handleNavClick(event) {
        event.preventDefault();
        const href = event.target.getAttribute('href');
        
        if (href.startsWith('#')) {
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

    showLoading() {
        this.loadingOverlay.classList.add('active');
        document.body.style.overflow = 'hidden';
    }

    hideLoading() {
        this.loadingOverlay.classList.remove('active');
        document.body.style.overflow = '';
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
            btn.addEventListener('click', () => {
                this.logEvent('download_attempt', {
                    button_id: btn.id || 'unknown',
                    timestamp: new Date().toISOString()
                });
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