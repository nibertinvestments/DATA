# Basic React Dataset - Component-Based UI Development

## Dataset 1: Basic Components and JSX
```jsx
import React from 'react';

// Functional component with JSX
function Welcome() {
    return <h1>Hello, World!</h1>;
}

// Component with props
function Greeting({ name, age }) {
    return (
        <div>
            <h2>Hello, {name}!</h2>
            <p>You are {age} years old.</p>
        </div>
    );
}

// Component with children
function Card({ children, title }) {
    return (
        <div className="card">
            <h3 className="card-title">{title}</h3>
            <div className="card-content">
                {children}
            </div>
        </div>
    );
}

// JSX expressions and conditionals
function UserProfile({ user, isLoggedIn }) {
    const userStatus = isLoggedIn ? 'online' : 'offline';
    
    return (
        <div className={`user-profile ${userStatus}`}>
            {isLoggedIn ? (
                <div>
                    <img src={user.avatar} alt={`${user.name} avatar`} />
                    <h2>{user.name}</h2>
                    <p>Email: {user.email}</p>
                </div>
            ) : (
                <p>Please log in to view profile</p>
            )}
        </div>
    );
}

// Component composition
function App() {
    const user = {
        name: 'Alice Smith',
        email: 'alice@example.com',
        avatar: '/avatars/alice.jpg'
    };
    
    return (
        <div className="app">
            <Welcome />
            <Greeting name="React Developer" age={25} />
            
            <Card title="User Information">
                <UserProfile user={user} isLoggedIn={true} />
            </Card>
            
            <Card title="Getting Started">
                <p>Welcome to React development!</p>
                <ul>
                    <li>Create components</li>
                    <li>Pass props</li>
                    <li>Handle events</li>
                </ul>
            </Card>
        </div>
    );
}

export default App;
```

## Dataset 2: State Management with Hooks
```jsx
import React, { useState, useEffect } from 'react';

// Basic useState hook
function Counter() {
    const [count, setCount] = useState(0);
    
    const increment = () => setCount(count + 1);
    const decrement = () => setCount(count - 1);
    const reset = () => setCount(0);
    
    return (
        <div className="counter">
            <h3>Counter: {count}</h3>
            <div className="counter-controls">
                <button onClick={decrement}>-</button>
                <button onClick={reset}>Reset</button>
                <button onClick={increment}>+</button>
            </div>
        </div>
    );
}

// Form state management
function ContactForm() {
    const [formData, setFormData] = useState({
        name: '',
        email: '',
        message: ''
    });
    
    const [errors, setErrors] = useState({});
    const [isSubmitting, setIsSubmitting] = useState(false);
    
    const handleChange = (e) => {
        const { name, value } = e.target;
        setFormData(prev => ({
            ...prev,
            [name]: value
        }));
        
        // Clear error when user starts typing
        if (errors[name]) {
            setErrors(prev => ({
                ...prev,
                [name]: ''
            }));
        }
    };
    
    const validateForm = () => {
        const newErrors = {};
        
        if (!formData.name.trim()) {
            newErrors.name = 'Name is required';
        }
        
        if (!formData.email.trim()) {
            newErrors.email = 'Email is required';
        } else if (!/\S+@\S+\.\S+/.test(formData.email)) {
            newErrors.email = 'Email is invalid';
        }
        
        if (!formData.message.trim()) {
            newErrors.message = 'Message is required';
        }
        
        return newErrors;
    };
    
    const handleSubmit = async (e) => {
        e.preventDefault();
        
        const formErrors = validateForm();
        if (Object.keys(formErrors).length > 0) {
            setErrors(formErrors);
            return;
        }
        
        setIsSubmitting(true);
        
        try {
            // Simulate API call
            await new Promise(resolve => setTimeout(resolve, 2000));
            console.log('Form submitted:', formData);
            
            // Reset form on success
            setFormData({ name: '', email: '', message: '' });
            alert('Message sent successfully!');
        } catch (error) {
            alert('Error sending message');
        } finally {
            setIsSubmitting(false);
        }
    };
    
    return (
        <form onSubmit={handleSubmit} className="contact-form">
            <h3>Contact Us</h3>
            
            <div className="form-group">
                <label htmlFor="name">Name:</label>
                <input
                    type="text"
                    id="name"
                    name="name"
                    value={formData.name}
                    onChange={handleChange}
                    className={errors.name ? 'error' : ''}
                />
                {errors.name && <span className="error-message">{errors.name}</span>}
            </div>
            
            <div className="form-group">
                <label htmlFor="email">Email:</label>
                <input
                    type="email"
                    id="email"
                    name="email"
                    value={formData.email}
                    onChange={handleChange}
                    className={errors.email ? 'error' : ''}
                />
                {errors.email && <span className="error-message">{errors.email}</span>}
            </div>
            
            <div className="form-group">
                <label htmlFor="message">Message:</label>
                <textarea
                    id="message"
                    name="message"
                    value={formData.message}
                    onChange={handleChange}
                    rows="4"
                    className={errors.message ? 'error' : ''}
                />
                {errors.message && <span className="error-message">{errors.message}</span>}
            </div>
            
            <button type="submit" disabled={isSubmitting}>
                {isSubmitting ? 'Sending...' : 'Send Message'}
            </button>
        </form>
    );
}

// Todo list with complex state
function TodoList() {
    const [todos, setTodos] = useState([]);
    const [newTodo, setNewTodo] = useState('');
    const [filter, setFilter] = useState('all'); // all, active, completed
    
    const addTodo = () => {
        if (newTodo.trim()) {
            setTodos([
                ...todos,
                {
                    id: Date.now(),
                    text: newTodo.trim(),
                    completed: false,
                    createdAt: new Date()
                }
            ]);
            setNewTodo('');
        }
    };
    
    const toggleTodo = (id) => {
        setTodos(todos.map(todo =>
            todo.id === id
                ? { ...todo, completed: !todo.completed }
                : todo
        ));
    };
    
    const deleteTodo = (id) => {
        setTodos(todos.filter(todo => todo.id !== id));
    };
    
    const clearCompleted = () => {
        setTodos(todos.filter(todo => !todo.completed));
    };
    
    const filteredTodos = todos.filter(todo => {
        if (filter === 'active') return !todo.completed;
        if (filter === 'completed') return todo.completed;
        return true; // all
    });
    
    const completedCount = todos.filter(todo => todo.completed).length;
    const activeCount = todos.length - completedCount;
    
    return (
        <div className="todo-app">
            <h3>Todo List</h3>
            
            <div className="todo-input">
                <input
                    type="text"
                    value={newTodo}
                    onChange={(e) => setNewTodo(e.target.value)}
                    placeholder="What needs to be done?"
                    onKeyPress={(e) => e.key === 'Enter' && addTodo()}
                />
                <button onClick={addTodo}>Add</button>
            </div>
            
            <div className="todo-filters">
                <button
                    className={filter === 'all' ? 'active' : ''}
                    onClick={() => setFilter('all')}
                >
                    All ({todos.length})
                </button>
                <button
                    className={filter === 'active' ? 'active' : ''}
                    onClick={() => setFilter('active')}
                >
                    Active ({activeCount})
                </button>
                <button
                    className={filter === 'completed' ? 'active' : ''}
                    onClick={() => setFilter('completed')}
                >
                    Completed ({completedCount})
                </button>
            </div>
            
            <ul className="todo-list">
                {filteredTodos.map(todo => (
                    <li key={todo.id} className={todo.completed ? 'completed' : ''}>
                        <input
                            type="checkbox"
                            checked={todo.completed}
                            onChange={() => toggleTodo(todo.id)}
                        />
                        <span className="todo-text">{todo.text}</span>
                        <button
                            className="delete-btn"
                            onClick={() => deleteTodo(todo.id)}
                        >
                            ×
                        </button>
                    </li>
                ))}
            </ul>
            
            {completedCount > 0 && (
                <button onClick={clearCompleted} className="clear-completed">
                    Clear Completed
                </button>
            )}
        </div>
    );
}

export { Counter, ContactForm, TodoList };
```

## Dataset 3: useEffect and Side Effects
```jsx
import React, { useState, useEffect, useRef } from 'react';

// Basic useEffect - component lifecycle
function Timer() {
    const [seconds, setSeconds] = useState(0);
    const [isRunning, setIsRunning] = useState(false);
    
    useEffect(() => {
        let interval = null;
        
        if (isRunning) {
            interval = setInterval(() => {
                setSeconds(seconds => seconds + 1);
            }, 1000);
        } else if (!isRunning && seconds !== 0) {
            clearInterval(interval);
        }
        
        // Cleanup function
        return () => clearInterval(interval);
    }, [isRunning, seconds]);
    
    const start = () => setIsRunning(true);
    const stop = () => setIsRunning(false);
    const reset = () => {
        setSeconds(0);
        setIsRunning(false);
    };
    
    return (
        <div className="timer">
            <h3>Timer: {seconds}s</h3>
            <div className="timer-controls">
                <button onClick={start} disabled={isRunning}>Start</button>
                <button onClick={stop} disabled={!isRunning}>Stop</button>
                <button onClick={reset}>Reset</button>
            </div>
        </div>
    );
}

// Data fetching with useEffect
function UsersList() {
    const [users, setUsers] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    
    useEffect(() => {
        const fetchUsers = async () => {
            try {
                setLoading(true);
                const response = await fetch('https://jsonplaceholder.typicode.com/users');
                
                if (!response.ok) {
                    throw new Error('Failed to fetch users');
                }
                
                const userData = await response.json();
                setUsers(userData);
            } catch (err) {
                setError(err.message);
            } finally {
                setLoading(false);
            }
        };
        
        fetchUsers();
    }, []); // Empty dependency array - run once on mount
    
    if (loading) return <div className="loading">Loading users...</div>;
    if (error) return <div className="error">Error: {error}</div>;
    
    return (
        <div className="users-list">
            <h3>Users</h3>
            <ul>
                {users.map(user => (
                    <li key={user.id} className="user-item">
                        <h4>{user.name}</h4>
                        <p>Email: {user.email}</p>
                        <p>Company: {user.company.name}</p>
                    </li>
                ))}
            </ul>
        </div>
    );
}

// Window resize listener
function WindowSize() {
    const [windowSize, setWindowSize] = useState({
        width: window.innerWidth,
        height: window.innerHeight
    });
    
    useEffect(() => {
        const handleResize = () => {
            setWindowSize({
                width: window.innerWidth,
                height: window.innerHeight
            });
        };
        
        window.addEventListener('resize', handleResize);
        
        // Cleanup
        return () => window.removeEventListener('resize', handleResize);
    }, []); // No dependencies - set up once
    
    return (
        <div className="window-size">
            <h3>Window Size</h3>
            <p>Width: {windowSize.width}px</p>
            <p>Height: {windowSize.height}px</p>
        </div>
    );
}

// Document title updater
function DocumentTitle({ title }) {
    useEffect(() => {
        const originalTitle = document.title;
        document.title = title;
        
        // Cleanup - restore original title
        return () => {
            document.title = originalTitle;
        };
    }, [title]);
    
    return null; // This component doesn't render anything
}

// Mouse position tracker
function MouseTracker() {
    const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
    const [isTracking, setIsTracking] = useState(false);
    
    useEffect(() => {
        if (!isTracking) return;
        
        const handleMouseMove = (event) => {
            setMousePosition({
                x: event.clientX,
                y: event.clientY
            });
        };
        
        document.addEventListener('mousemove', handleMouseMove);
        
        return () => {
            document.removeEventListener('mousemove', handleMouseMove);
        };
    }, [isTracking]);
    
    return (
        <div className="mouse-tracker">
            <h3>Mouse Tracker</h3>
            <button onClick={() => setIsTracking(!isTracking)}>
                {isTracking ? 'Stop Tracking' : 'Start Tracking'}
            </button>
            
            {isTracking && (
                <div className="mouse-position">
                    Mouse Position: X: {mousePosition.x}, Y: {mousePosition.y}
                </div>
            )}
        </div>
    );
}

// Local storage synchronization
function useLocalStorage(key, initialValue) {
    // Get initial value from localStorage or use provided initial value
    const [storedValue, setStoredValue] = useState(() => {
        try {
            const item = localStorage.getItem(key);
            return item ? JSON.parse(item) : initialValue;
        } catch (error) {
            console.error('Error reading from localStorage:', error);
            return initialValue;
        }
    });
    
    // Update localStorage when state changes
    const setValue = (value) => {
        try {
            const valueToStore = value instanceof Function ? value(storedValue) : value;
            setStoredValue(valueToStore);
            localStorage.setItem(key, JSON.stringify(valueToStore));
        } catch (error) {
            console.error('Error writing to localStorage:', error);
        }
    };
    
    return [storedValue, setValue];
}

function Settings() {
    const [theme, setTheme] = useLocalStorage('theme', 'light');
    const [fontSize, setFontSize] = useLocalStorage('fontSize', 16);
    const [notifications, setNotifications] = useLocalStorage('notifications', true);
    
    useEffect(() => {
        // Apply theme to document
        document.body.className = theme;
        document.body.style.fontSize = `${fontSize}px`;
    }, [theme, fontSize]);
    
    return (
        <div className="settings">
            <h3>Settings</h3>
            
            <div className="setting-item">
                <label>Theme:</label>
                <select value={theme} onChange={(e) => setTheme(e.target.value)}>
                    <option value="light">Light</option>
                    <option value="dark">Dark</option>
                </select>
            </div>
            
            <div className="setting-item">
                <label>Font Size:</label>
                <input
                    type="range"
                    min="12"
                    max="24"
                    value={fontSize}
                    onChange={(e) => setFontSize(parseInt(e.target.value))}
                />
                <span>{fontSize}px</span>
            </div>
            
            <div className="setting-item">
                <label>
                    <input
                        type="checkbox"
                        checked={notifications}
                        onChange={(e) => setNotifications(e.target.checked)}
                    />
                    Enable Notifications
                </label>
            </div>
        </div>
    );
}

export { Timer, UsersList, WindowSize, DocumentTitle, MouseTracker, Settings };
```

## Dataset 4: Event Handling and Forms
```jsx
import React, { useState, useRef, useCallback } from 'react';

// Basic event handling
function ButtonEvents() {
    const [clickCount, setClickCount] = useState(0);
    const [lastEvent, setLastEvent] = useState('');
    
    const handleClick = (event) => {
        setClickCount(prev => prev + 1);
        setLastEvent(`Click at (${event.clientX}, ${event.clientY})`);
    };
    
    const handleMouseEnter = () => {
        setLastEvent('Mouse entered');
    };
    
    const handleMouseLeave = () => {
        setLastEvent('Mouse left');
    };
    
    const handleDoubleClick = () => {
        setLastEvent('Double clicked');
        setClickCount(0); // Reset on double click
    };
    
    return (
        <div className="button-events">
            <h3>Button Events</h3>
            <button
                onClick={handleClick}
                onMouseEnter={handleMouseEnter}
                onMouseLeave={handleMouseLeave}
                onDoubleClick={handleDoubleClick}
                className="event-button"
            >
                Click me! (Clicked {clickCount} times)
            </button>
            <p>Last event: {lastEvent}</p>
        </div>
    );
}

// Advanced form with validation
function RegistrationForm() {
    const [formData, setFormData] = useState({
        username: '',
        email: '',
        password: '',
        confirmPassword: '',
        age: '',
        interests: [],
        newsletter: false,
        country: ''
    });
    
    const [errors, setErrors] = useState({});
    const [touched, setTouched] = useState({});
    const usernameRef = useRef(null);
    
    // Focus username field on component mount
    React.useEffect(() => {
        usernameRef.current?.focus();
    }, []);
    
    const handleInputChange = useCallback((event) => {
        const { name, value, type, checked } = event.target;
        
        setFormData(prev => ({
            ...prev,
            [name]: type === 'checkbox' ? checked : value
        }));
        
        // Clear error when user starts typing
        if (errors[name]) {
            setErrors(prev => ({ ...prev, [name]: '' }));
        }
    }, [errors]);
    
    const handleInterestChange = useCallback((interest) => {
        setFormData(prev => ({
            ...prev,
            interests: prev.interests.includes(interest)
                ? prev.interests.filter(i => i !== interest)
                : [...prev.interests, interest]
        }));
    }, []);
    
    const handleBlur = useCallback((event) => {
        const { name } = event.target;
        setTouched(prev => ({ ...prev, [name]: true }));
        validateField(name, formData[name]);
    }, [formData]);
    
    const validateField = (name, value) => {
        let error = '';
        
        switch (name) {
            case 'username':
                if (!value.trim()) {
                    error = 'Username is required';
                } else if (value.length < 3) {
                    error = 'Username must be at least 3 characters';
                } else if (!/^[a-zA-Z0-9_]+$/.test(value)) {
                    error = 'Username can only contain letters, numbers, and underscores';
                }
                break;
                
            case 'email':
                if (!value.trim()) {
                    error = 'Email is required';
                } else if (!/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value)) {
                    error = 'Please enter a valid email address';
                }
                break;
                
            case 'password':
                if (!value) {
                    error = 'Password is required';
                } else if (value.length < 6) {
                    error = 'Password must be at least 6 characters';
                } else if (!/(?=.*[a-z])(?=.*[A-Z])(?=.*\d)/.test(value)) {
                    error = 'Password must contain at least one lowercase, uppercase, and number';
                }
                break;
                
            case 'confirmPassword':
                if (!value) {
                    error = 'Please confirm your password';
                } else if (value !== formData.password) {
                    error = 'Passwords do not match';
                }
                break;
                
            case 'age':
                const ageNum = parseInt(value);
                if (!value) {
                    error = 'Age is required';
                } else if (isNaN(ageNum) || ageNum < 13 || ageNum > 120) {
                    error = 'Please enter a valid age between 13 and 120';
                }
                break;
                
            default:
                break;
        }
        
        setErrors(prev => ({ ...prev, [name]: error }));
        return error === '';
    };
    
    const validateForm = () => {
        const fields = ['username', 'email', 'password', 'confirmPassword', 'age'];
        let isValid = true;
        
        fields.forEach(field => {
            const fieldIsValid = validateField(field, formData[field]);
            if (!fieldIsValid) isValid = false;
        });
        
        return isValid;
    };
    
    const handleSubmit = (event) => {
        event.preventDefault();
        
        // Mark all fields as touched
        const allFields = ['username', 'email', 'password', 'confirmPassword', 'age'];
        setTouched(prev => allFields.reduce((acc, field) => ({ ...acc, [field]: true }), prev));
        
        if (validateForm()) {
            console.log('Registration data:', formData);
            alert('Registration successful!');
            
            // Reset form
            setFormData({
                username: '',
                email: '',
                password: '',
                confirmPassword: '',
                age: '',
                interests: [],
                newsletter: false,
                country: ''
            });
            setErrors({});
            setTouched({});
        }
    };
    
    const interestOptions = ['Programming', 'Music', 'Sports', 'Reading', 'Travel', 'Gaming'];
    const countryOptions = ['', 'USA', 'Canada', 'UK', 'Germany', 'France', 'Australia', 'Other'];
    
    return (
        <form onSubmit={handleSubmit} className="registration-form">
            <h3>Registration Form</h3>
            
            <div className="form-group">
                <label htmlFor="username">Username*</label>
                <input
                    ref={usernameRef}
                    type="text"
                    id="username"
                    name="username"
                    value={formData.username}
                    onChange={handleInputChange}
                    onBlur={handleBlur}
                    className={errors.username && touched.username ? 'error' : ''}
                    placeholder="Enter username"
                />
                {errors.username && touched.username && (
                    <span className="error-message">{errors.username}</span>
                )}
            </div>
            
            <div className="form-group">
                <label htmlFor="email">Email*</label>
                <input
                    type="email"
                    id="email"
                    name="email"
                    value={formData.email}
                    onChange={handleInputChange}
                    onBlur={handleBlur}
                    className={errors.email && touched.email ? 'error' : ''}
                    placeholder="Enter email"
                />
                {errors.email && touched.email && (
                    <span className="error-message">{errors.email}</span>
                )}
            </div>
            
            <div className="form-group">
                <label htmlFor="password">Password*</label>
                <input
                    type="password"
                    id="password"
                    name="password"
                    value={formData.password}
                    onChange={handleInputChange}
                    onBlur={handleBlur}
                    className={errors.password && touched.password ? 'error' : ''}
                    placeholder="Enter password"
                />
                {errors.password && touched.password && (
                    <span className="error-message">{errors.password}</span>
                )}
            </div>
            
            <div className="form-group">
                <label htmlFor="confirmPassword">Confirm Password*</label>
                <input
                    type="password"
                    id="confirmPassword"
                    name="confirmPassword"
                    value={formData.confirmPassword}
                    onChange={handleInputChange}
                    onBlur={handleBlur}
                    className={errors.confirmPassword && touched.confirmPassword ? 'error' : ''}
                    placeholder="Confirm password"
                />
                {errors.confirmPassword && touched.confirmPassword && (
                    <span className="error-message">{errors.confirmPassword}</span>
                )}
            </div>
            
            <div className="form-group">
                <label htmlFor="age">Age*</label>
                <input
                    type="number"
                    id="age"
                    name="age"
                    value={formData.age}
                    onChange={handleInputChange}
                    onBlur={handleBlur}
                    className={errors.age && touched.age ? 'error' : ''}
                    placeholder="Enter age"
                    min="13"
                    max="120"
                />
                {errors.age && touched.age && (
                    <span className="error-message">{errors.age}</span>
                )}
            </div>
            
            <div className="form-group">
                <label>Country</label>
                <select
                    name="country"
                    value={formData.country}
                    onChange={handleInputChange}
                >
                    {countryOptions.map(country => (
                        <option key={country} value={country}>
                            {country || 'Select a country'}
                        </option>
                    ))}
                </select>
            </div>
            
            <div className="form-group">
                <label>Interests</label>
                <div className="checkbox-group">
                    {interestOptions.map(interest => (
                        <label key={interest} className="checkbox-label">
                            <input
                                type="checkbox"
                                checked={formData.interests.includes(interest)}
                                onChange={() => handleInterestChange(interest)}
                            />
                            {interest}
                        </label>
                    ))}
                </div>
            </div>
            
            <div className="form-group">
                <label className="checkbox-label">
                    <input
                        type="checkbox"
                        name="newsletter"
                        checked={formData.newsletter}
                        onChange={handleInputChange}
                    />
                    Subscribe to newsletter
                </label>
            </div>
            
            <button type="submit" className="submit-button">
                Register
            </button>
        </form>
    );
}

// File upload component
function FileUpload() {
    const [selectedFiles, setSelectedFiles] = useState([]);
    const [dragOver, setDragOver] = useState(false);
    const fileInputRef = useRef(null);
    
    const handleFileSelect = (files) => {
        const fileArray = Array.from(files);
        const fileData = fileArray.map(file => ({
            file,
            name: file.name,
            size: file.size,
            type: file.type,
            id: Math.random().toString(36).substr(2, 9)
        }));
        
        setSelectedFiles(prev => [...prev, ...fileData]);
    };
    
    const handleFileInputChange = (event) => {
        handleFileSelect(event.target.files);
    };
    
    const handleDragOver = (event) => {
        event.preventDefault();
        setDragOver(true);
    };
    
    const handleDragLeave = (event) => {
        event.preventDefault();
        setDragOver(false);
    };
    
    const handleDrop = (event) => {
        event.preventDefault();
        setDragOver(false);
        handleFileSelect(event.dataTransfer.files);
    };
    
    const removeFile = (id) => {
        setSelectedFiles(prev => prev.filter(file => file.id !== id));
    };
    
    const formatFileSize = (bytes) => {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    };
    
    return (
        <div className="file-upload">
            <h3>File Upload</h3>
            
            <div
                className={`drop-zone ${dragOver ? 'drag-over' : ''}`}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
            >
                <p>Drag & drop files here or click to select</p>
                <input
                    ref={fileInputRef}
                    type="file"
                    multiple
                    onChange={handleFileInputChange}
                    style={{ display: 'none' }}
                />
            </div>
            
            {selectedFiles.length > 0 && (
                <div className="selected-files">
                    <h4>Selected Files:</h4>
                    <ul>
                        {selectedFiles.map(({ id, name, size, type }) => (
                            <li key={id} className="file-item">
                                <span className="file-name">{name}</span>
                                <span className="file-size">({formatFileSize(size)})</span>
                                <span className="file-type">{type}</span>
                                <button
                                    className="remove-button"
                                    onClick={() => removeFile(id)}
                                >
                                    Remove
                                </button>
                            </li>
                        ))}
                    </ul>
                </div>
            )}
        </div>
    );
}

export { ButtonEvents, RegistrationForm, FileUpload };
```