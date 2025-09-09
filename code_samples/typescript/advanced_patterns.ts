// Comprehensive TypeScript Advanced Development Examples
// Demonstrates modern TypeScript features, advanced types, and full-stack patterns

import { EventEmitter } from 'events';
import { v4 as uuidv4 } from 'uuid';

// ============ Advanced Type System ============

// Utility types and type manipulation
type NonNullable<T> = T extends null | undefined ? never : T;
type DeepReadonly<T> = {
  readonly [P in keyof T]: T[P] extends object ? DeepReadonly<T[P]> : T[P];
};

type DeepPartial<T> = {
  [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P];
};

// Advanced mapped types
type ExtractFunctions<T> = {
  [K in keyof T]: T[K] extends (...args: any[]) => any ? T[K] : never;
};

type FunctionPropertyNames<T> = {
  [K in keyof T]: T[K] extends (...args: any[]) => any ? K : never;
}[keyof T];

// Conditional types with inference
type ReturnTypeOf<T> = T extends (...args: any[]) => infer R ? R : never;
type ParametersOf<T> = T extends (...args: infer P) => any ? P : never;

// Template literal types
type EventName<T extends string> = `on${Capitalize<T>}`;
type CSSProperty = `--${string}`;
type APIEndpoint<T extends string> = `/api/${T}`;

// Branded types for type safety
declare const __brand: unique symbol;
type Brand<T, B> = T & { [__brand]: B };

type UserId = Brand<string, 'UserId'>;
type Email = Brand<string, 'Email'>;
type Password = Brand<string, 'Password'>;
type UUID = Brand<string, 'UUID'>;

// ============ Domain Models with Rich Types ============

// Enums and union types
enum UserRole {
  GUEST = 'guest',
  USER = 'user',
  MODERATOR = 'moderator',
  ADMIN = 'admin',
  SUPER_ADMIN = 'super_admin'
}

enum TaskStatus {
  DRAFT = 'draft',
  ACTIVE = 'active',
  IN_PROGRESS = 'in_progress',
  REVIEW = 'review',
  COMPLETED = 'completed',
  CANCELLED = 'cancelled',
  ARCHIVED = 'archived'
}

enum Priority {
  LOW = 1,
  MEDIUM = 2,
  HIGH = 3,
  CRITICAL = 4
}

// Discriminated unions for state management
type LoadingState = {
  status: 'loading';
  progress?: number;
};

type SuccessState<T> = {
  status: 'success';
  data: T;
};

type ErrorState = {
  status: 'error';
  error: string;
  code?: number;
};

type IdleState = {
  status: 'idle';
};

type AsyncState<T> = LoadingState | SuccessState<T> | ErrorState | IdleState;

// Complex interface hierarchies
interface Timestamped {
  readonly createdAt: Date;
  readonly updatedAt: Date;
}

interface Identifiable<T = string> {
  readonly id: T;
}

interface Auditable extends Timestamped {
  readonly createdBy: UserId;
  readonly updatedBy: UserId;
  readonly version: number;
}

interface SoftDeletable {
  readonly deletedAt?: Date;
  readonly isDeleted: boolean;
}

interface Searchable {
  readonly searchableText: string;
}

// Generic constraints and complex types
interface Repository<T extends Identifiable, K = string> {
  findById(id: K): Promise<T | null>;
  findAll(options?: QueryOptions<T>): Promise<PaginatedResult<T>>;
  create(entity: Omit<T, 'id' | keyof Timestamped>): Promise<T>;
  update(id: K, updates: DeepPartial<Omit<T, 'id' | 'createdAt'>>): Promise<T>;
  delete(id: K): Promise<boolean>;
  count(filter?: FilterCriteria<T>): Promise<number>;
}

type QueryOptions<T> = {
  filter?: FilterCriteria<T>;
  sort?: SortCriteria<T>;
  pagination?: PaginationOptions;
  include?: (keyof T)[];
};

type FilterCriteria<T> = {
  [K in keyof T]?: T[K] extends string
    ? StringFilterOptions
    : T[K] extends number
    ? NumberFilterOptions
    : T[K] extends Date
    ? DateFilterOptions
    : T[K] extends boolean
    ? boolean
    : any;
};

type StringFilterOptions = {
  equals?: string;
  contains?: string;
  startsWith?: string;
  endsWith?: string;
  in?: string[];
  regex?: RegExp;
};

type NumberFilterOptions = {
  equals?: number;
  gt?: number;
  gte?: number;
  lt?: number;
  lte?: number;
  in?: number[];
};

type DateFilterOptions = {
  equals?: Date;
  after?: Date;
  before?: Date;
  between?: [Date, Date];
};

type SortCriteria<T> = {
  [K in keyof T]?: 'asc' | 'desc';
};

type PaginationOptions = {
  page: number;
  limit: number;
};

type PaginatedResult<T> = {
  data: T[];
  pagination: {
    page: number;
    limit: number;
    total: number;
    totalPages: number;
    hasNext: boolean;
    hasPrev: boolean;
  };
};

// Value objects with validation
class EmailAddress {
  private constructor(private readonly value: string) {}

  static create(email: string): Result<EmailAddress, ValidationError> {
    if (!email || typeof email !== 'string') {
      return Result.failure(new ValidationError('Email is required'));
    }

    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(email)) {
      return Result.failure(new ValidationError('Invalid email format'));
    }

    return Result.success(new EmailAddress(email.toLowerCase().trim()));
  }

  getValue(): Email {
    return this.value as Email;
  }

  getDomain(): string {
    return this.value.split('@')[1];
  }

  getLocalPart(): string {
    return this.value.split('@')[0];
  }

  equals(other: EmailAddress): boolean {
    return this.value === other.value;
  }

  toString(): string {
    return this.value;
  }
}

class Money {
  private constructor(
    private readonly amount: number,
    private readonly currency: string
  ) {}

  static create(amount: number, currency: string = 'USD'): Result<Money, ValidationError> {
    if (typeof amount !== 'number' || isNaN(amount)) {
      return Result.failure(new ValidationError('Amount must be a valid number'));
    }

    if (amount < 0) {
      return Result.failure(new ValidationError('Amount cannot be negative'));
    }

    if (!currency || typeof currency !== 'string') {
      return Result.failure(new ValidationError('Currency is required'));
    }

    return Result.success(new Money(Math.round(amount * 100) / 100, currency.toUpperCase()));
  }

  add(other: Money): Result<Money, DomainError> {
    if (this.currency !== other.currency) {
      return Result.failure(new DomainError('Cannot add money with different currencies'));
    }

    return Money.create(this.amount + other.amount, this.currency);
  }

  subtract(other: Money): Result<Money, DomainError> {
    if (this.currency !== other.currency) {
      return Result.failure(new DomainError('Cannot subtract money with different currencies'));
    }

    const result = this.amount - other.amount;
    if (result < 0) {
      return Result.failure(new DomainError('Result cannot be negative'));
    }

    return Money.create(result, this.currency);
  }

  multiply(factor: number): Result<Money, ValidationError> {
    if (typeof factor !== 'number' || isNaN(factor) || factor < 0) {
      return Result.failure(new ValidationError('Factor must be a positive number'));
    }

    return Money.create(this.amount * factor, this.currency);
  }

  getAmount(): number {
    return this.amount;
  }

  getCurrency(): string {
    return this.currency;
  }

  equals(other: Money): boolean {
    return this.amount === other.amount && this.currency === other.currency;
  }

  toString(): string {
    return `${this.amount.toFixed(2)} ${this.currency}`;
  }
}

// ============ Result Type for Error Handling ============

class Result<T, E = Error> {
  private constructor(
    private readonly isSuccess: boolean,
    private readonly value?: T,
    private readonly error?: E
  ) {}

  static success<T, E = Error>(value: T): Result<T, E> {
    return new Result<T, E>(true, value);
  }

  static failure<T, E = Error>(error: E): Result<T, E> {
    return new Result<T, E>(false, undefined, error);
  }

  isOk(): this is Result<T, never> {
    return this.isSuccess;
  }

  isErr(): this is Result<never, E> {
    return !this.isSuccess;
  }

  unwrap(): T {
    if (!this.isSuccess) {
      throw new Error('Called unwrap on a failure result');
    }
    return this.value!;
  }

  unwrapOr(defaultValue: T): T {
    return this.isSuccess ? this.value! : defaultValue;
  }

  unwrapErr(): E {
    if (this.isSuccess) {
      throw new Error('Called unwrapErr on a success result');
    }
    return this.error!;
  }

  map<U>(fn: (value: T) => U): Result<U, E> {
    if (this.isSuccess) {
      return Result.success(fn(this.value!));
    }
    return Result.failure(this.error!);
  }

  mapErr<F>(fn: (error: E) => F): Result<T, F> {
    if (this.isSuccess) {
      return Result.success(this.value!);
    }
    return Result.failure(fn(this.error!));
  }

  flatMap<U>(fn: (value: T) => Result<U, E>): Result<U, E> {
    if (this.isSuccess) {
      return fn(this.value!);
    }
    return Result.failure(this.error!);
  }

  match<U>(onSuccess: (value: T) => U, onFailure: (error: E) => U): U {
    return this.isSuccess ? onSuccess(this.value!) : onFailure(this.error!);
  }

  async matchAsync<U>(
    onSuccess: (value: T) => Promise<U>,
    onFailure: (error: E) => Promise<U>
  ): Promise<U> {
    return this.isSuccess ? onSuccess(this.value!) : onFailure(this.error!);
  }
}

// ============ Custom Error Classes ============

abstract class DomainError extends Error {
  abstract readonly code: string;
  
  constructor(message: string) {
    super(message);
    this.name = this.constructor.name;
  }
}

class ValidationError extends DomainError {
  readonly code = 'VALIDATION_ERROR';
  
  constructor(message: string, public readonly field?: string) {
    super(message);
  }
}

class NotFoundError extends DomainError {
  readonly code = 'NOT_FOUND';
  
  constructor(resource: string, id: string) {
    super(`${resource} with id '${id}' not found`);
  }
}

class UnauthorizedError extends DomainError {
  readonly code = 'UNAUTHORIZED';
  
  constructor(message: string = 'Unauthorized access') {
    super(message);
  }
}

class BusinessRuleError extends DomainError {
  readonly code = 'BUSINESS_RULE_VIOLATION';
  
  constructor(rule: string, message: string) {
    super(`Business rule violation: ${rule} - ${message}`);
  }
}

// ============ Domain Entities ============

interface UserProps extends Identifiable<UserId>, Auditable {
  firstName: string;
  lastName: string;
  email: EmailAddress;
  role: UserRole;
  isActive: boolean;
  lastLoginAt?: Date;
  preferences: UserPreferences;
  profile: UserProfile;
}

interface UserPreferences {
  theme: 'light' | 'dark' | 'auto';
  language: string;
  timezone: string;
  notifications: {
    email: boolean;
    push: boolean;
    sms: boolean;
  };
}

interface UserProfile {
  bio?: string;
  avatar?: string;
  dateOfBirth?: Date;
  location?: string;
  website?: string;
  socialLinks: Record<string, string>;
}

class User implements UserProps {
  readonly id: UserId;
  readonly createdAt: Date;
  readonly updatedAt: Date;
  readonly createdBy: UserId;
  readonly updatedBy: UserId;
  readonly version: number;
  
  constructor(
    public firstName: string,
    public lastName: string,
    public email: EmailAddress,
    public role: UserRole,
    public isActive: boolean,
    public preferences: UserPreferences,
    public profile: UserProfile,
    public lastLoginAt?: Date,
    id?: UserId
  ) {
    this.id = id || (uuidv4() as UserId);
    const now = new Date();
    this.createdAt = now;
    this.updatedAt = now;
    this.createdBy = this.id;
    this.updatedBy = this.id;
    this.version = 1;
  }

  static create(props: {
    firstName: string;
    lastName: string;
    email: string;
    role?: UserRole;
    preferences?: Partial<UserPreferences>;
    profile?: Partial<UserProfile>;
  }): Result<User, ValidationError> {
    // Validate required fields
    if (!props.firstName?.trim()) {
      return Result.failure(new ValidationError('First name is required', 'firstName'));
    }

    if (!props.lastName?.trim()) {
      return Result.failure(new ValidationError('Last name is required', 'lastName'));
    }

    // Create email value object
    const emailResult = EmailAddress.create(props.email);
    if (emailResult.isErr()) {
      return Result.failure(emailResult.unwrapErr());
    }

    // Set defaults
    const defaultPreferences: UserPreferences = {
      theme: 'auto',
      language: 'en',
      timezone: 'UTC',
      notifications: {
        email: true,
        push: true,
        sms: false
      }
    };

    const defaultProfile: UserProfile = {
      socialLinks: {}
    };

    const user = new User(
      props.firstName.trim(),
      props.lastName.trim(),
      emailResult.unwrap(),
      props.role || UserRole.USER,
      true,
      { ...defaultPreferences, ...props.preferences },
      { ...defaultProfile, ...props.profile }
    );

    return Result.success(user);
  }

  updateProfile(updates: Partial<UserProfile>): Result<User, ValidationError> {
    const newProfile = { ...this.profile, ...updates };
    
    // Validation logic here
    if (updates.website && !this.isValidUrl(updates.website)) {
      return Result.failure(new ValidationError('Invalid website URL', 'website'));
    }

    this.profile = newProfile;
    return Result.success(this);
  }

  changeRole(newRole: UserRole, changedBy: UserId): Result<User, BusinessRuleError> {
    if (this.role === UserRole.SUPER_ADMIN && newRole !== UserRole.SUPER_ADMIN) {
      return Result.failure(
        new BusinessRuleError('SUPER_ADMIN_DEMOTION', 'Super admin role cannot be changed')
      );
    }

    this.role = newRole;
    return Result.success(this);
  }

  deactivate(): Result<User, BusinessRuleError> {
    if (!this.isActive) {
      return Result.failure(
        new BusinessRuleError('ALREADY_INACTIVE', 'User is already inactive')
      );
    }

    this.isActive = false;
    return Result.success(this);
  }

  recordLogin(): User {
    this.lastLoginAt = new Date();
    return this;
  }

  get fullName(): string {
    return `${this.firstName} ${this.lastName}`;
  }

  get displayName(): string {
    return this.profile.bio ? `${this.fullName} - ${this.profile.bio}` : this.fullName;
  }

  hasPermission(permission: string): boolean {
    const rolePermissions: Record<UserRole, string[]> = {
      [UserRole.GUEST]: ['read:public'],
      [UserRole.USER]: ['read:public', 'read:own', 'write:own'],
      [UserRole.MODERATOR]: ['read:public', 'read:own', 'write:own', 'moderate:content'],
      [UserRole.ADMIN]: ['read:all', 'write:all', 'delete:all', 'manage:users'],
      [UserRole.SUPER_ADMIN]: ['*']
    };

    const permissions = rolePermissions[this.role] || [];
    return permissions.includes('*') || permissions.includes(permission);
  }

  private isValidUrl(url: string): boolean {
    try {
      new URL(url);
      return true;
    } catch {
      return false;
    }
  }
}

interface TaskProps extends Identifiable<number>, Auditable {
  title: string;
  description: string;
  status: TaskStatus;
  priority: Priority;
  assigneeId: UserId;
  dueDate?: Date;
  completedAt?: Date;
  estimatedHours?: number;
  actualHours?: number;
  tags: string[];
  attachments: TaskAttachment[];
  comments: TaskComment[];
}

interface TaskAttachment {
  id: string;
  fileName: string;
  fileSize: number;
  mimeType: string;
  url: string;
  uploadedAt: Date;
  uploadedBy: UserId;
}

interface TaskComment {
  id: string;
  content: string;
  authorId: UserId;
  createdAt: Date;
  updatedAt?: Date;
  isEdited: boolean;
}

class Task implements TaskProps {
  readonly id: number;
  readonly createdAt: Date;
  readonly updatedAt: Date;
  readonly createdBy: UserId;
  readonly updatedBy: UserId;
  readonly version: number;
  
  constructor(
    public title: string,
    public description: string,
    public status: TaskStatus,
    public priority: Priority,
    public assigneeId: UserId,
    createdBy: UserId,
    public tags: string[] = [],
    public attachments: TaskAttachment[] = [],
    public comments: TaskComment[] = [],
    public dueDate?: Date,
    public completedAt?: Date,
    public estimatedHours?: number,
    public actualHours?: number,
    id?: number
  ) {
    this.id = id || Math.floor(Math.random() * 1000000);
    const now = new Date();
    this.createdAt = now;
    this.updatedAt = now;
    this.createdBy = createdBy;
    this.updatedBy = createdBy;
    this.version = 1;
  }

  static create(props: {
    title: string;
    description: string;
    priority: Priority;
    assigneeId: UserId;
    createdBy: UserId;
    dueDate?: Date;
    estimatedHours?: number;
    tags?: string[];
  }): Result<Task, ValidationError> {
    // Validation
    if (!props.title?.trim()) {
      return Result.failure(new ValidationError('Title is required', 'title'));
    }

    if (props.title.length > 200) {
      return Result.failure(new ValidationError('Title cannot exceed 200 characters', 'title'));
    }

    if (props.description?.length > 2000) {
      return Result.failure(new ValidationError('Description cannot exceed 2000 characters', 'description'));
    }

    if (props.dueDate && props.dueDate < new Date()) {
      return Result.failure(new ValidationError('Due date cannot be in the past', 'dueDate'));
    }

    if (props.estimatedHours !== undefined && props.estimatedHours <= 0) {
      return Result.failure(new ValidationError('Estimated hours must be positive', 'estimatedHours'));
    }

    const task = new Task(
      props.title.trim(),
      props.description?.trim() || '',
      TaskStatus.DRAFT,
      props.priority,
      props.assigneeId,
      props.createdBy,
      props.tags || [],
      [],
      [],
      props.dueDate,
      undefined,
      props.estimatedHours
    );

    return Result.success(task);
  }

  updateDetails(updates: {
    title?: string;
    description?: string;
    priority?: Priority;
    dueDate?: Date;
    estimatedHours?: number;
  }): Result<Task, ValidationError> {
    // Validation logic similar to create
    if (updates.title !== undefined) {
      if (!updates.title.trim()) {
        return Result.failure(new ValidationError('Title is required', 'title'));
      }
      if (updates.title.length > 200) {
        return Result.failure(new ValidationError('Title cannot exceed 200 characters', 'title'));
      }
      this.title = updates.title.trim();
    }

    if (updates.description !== undefined) {
      if (updates.description.length > 2000) {
        return Result.failure(new ValidationError('Description cannot exceed 2000 characters', 'description'));
      }
      this.description = updates.description.trim();
    }

    if (updates.priority !== undefined) {
      this.priority = updates.priority;
    }

    if (updates.dueDate !== undefined) {
      if (updates.dueDate < new Date()) {
        return Result.failure(new ValidationError('Due date cannot be in the past', 'dueDate'));
      }
      this.dueDate = updates.dueDate;
    }

    if (updates.estimatedHours !== undefined) {
      if (updates.estimatedHours <= 0) {
        return Result.failure(new ValidationError('Estimated hours must be positive', 'estimatedHours'));
      }
      this.estimatedHours = updates.estimatedHours;
    }

    return Result.success(this);
  }

  changeStatus(newStatus: TaskStatus): Result<Task, BusinessRuleError> {
    const validTransitions = this.getValidStatusTransitions();
    
    if (!validTransitions.includes(newStatus)) {
      return Result.failure(
        new BusinessRuleError(
          'INVALID_STATUS_TRANSITION',
          `Cannot transition from ${this.status} to ${newStatus}`
        )
      );
    }

    this.status = newStatus;
    
    if (newStatus === TaskStatus.COMPLETED) {
      this.completedAt = new Date();
    } else if (this.status === TaskStatus.COMPLETED) {
      this.completedAt = undefined;
    }

    return Result.success(this);
  }

  addTag(tag: string): Result<Task, ValidationError> {
    const cleanTag = tag.trim().toLowerCase();
    
    if (!cleanTag) {
      return Result.failure(new ValidationError('Tag cannot be empty', 'tag'));
    }

    if (this.tags.includes(cleanTag)) {
      return Result.failure(new ValidationError('Tag already exists', 'tag'));
    }

    this.tags.push(cleanTag);
    return Result.success(this);
  }

  removeTag(tag: string): Result<Task, ValidationError> {
    const index = this.tags.indexOf(tag.trim().toLowerCase());
    
    if (index === -1) {
      return Result.failure(new ValidationError('Tag not found', 'tag'));
    }

    this.tags.splice(index, 1);
    return Result.success(this);
  }

  addComment(content: string, authorId: UserId): Result<Task, ValidationError> {
    if (!content?.trim()) {
      return Result.failure(new ValidationError('Comment content is required', 'content'));
    }

    const comment: TaskComment = {
      id: uuidv4(),
      content: content.trim(),
      authorId,
      createdAt: new Date(),
      isEdited: false
    };

    this.comments.push(comment);
    return Result.success(this);
  }

  recordWorkTime(hours: number): Result<Task, ValidationError> {
    if (hours <= 0) {
      return Result.failure(new ValidationError('Hours must be positive', 'hours'));
    }

    this.actualHours = (this.actualHours || 0) + hours;
    return Result.success(this);
  }

  get isOverdue(): boolean {
    return Boolean(
      this.dueDate &&
      this.dueDate < new Date() &&
      this.status !== TaskStatus.COMPLETED &&
      this.status !== TaskStatus.CANCELLED
    );
  }

  get isCompleted(): boolean {
    return this.status === TaskStatus.COMPLETED;
  }

  get progressPercentage(): number {
    if (this.status === TaskStatus.COMPLETED) return 100;
    if (this.status === TaskStatus.DRAFT) return 0;
    
    const statusProgress: Record<TaskStatus, number> = {
      [TaskStatus.DRAFT]: 0,
      [TaskStatus.ACTIVE]: 10,
      [TaskStatus.IN_PROGRESS]: 50,
      [TaskStatus.REVIEW]: 80,
      [TaskStatus.COMPLETED]: 100,
      [TaskStatus.CANCELLED]: 0,
      [TaskStatus.ARCHIVED]: 100
    };

    return statusProgress[this.status] || 0;
  }

  get timeToCompletion(): number | null {
    if (!this.completedAt) return null;
    return this.completedAt.getTime() - this.createdAt.getTime();
  }

  private getValidStatusTransitions(): TaskStatus[] {
    const transitions: Record<TaskStatus, TaskStatus[]> = {
      [TaskStatus.DRAFT]: [TaskStatus.ACTIVE, TaskStatus.CANCELLED],
      [TaskStatus.ACTIVE]: [TaskStatus.IN_PROGRESS, TaskStatus.CANCELLED],
      [TaskStatus.IN_PROGRESS]: [TaskStatus.REVIEW, TaskStatus.ACTIVE, TaskStatus.CANCELLED],
      [TaskStatus.REVIEW]: [TaskStatus.COMPLETED, TaskStatus.IN_PROGRESS, TaskStatus.CANCELLED],
      [TaskStatus.COMPLETED]: [TaskStatus.ARCHIVED, TaskStatus.ACTIVE],
      [TaskStatus.CANCELLED]: [TaskStatus.ACTIVE],
      [TaskStatus.ARCHIVED]: []
    };

    return transitions[this.status] || [];
  }
}

// ============ Repository Implementations ============

class InMemoryUserRepository implements Repository<User, UserId> {
  private users = new Map<UserId, User>();

  async findById(id: UserId): Promise<User | null> {
    return this.users.get(id) || null;
  }

  async findAll(options?: QueryOptions<User>): Promise<PaginatedResult<User>> {
    let users = Array.from(this.users.values());

    // Apply filtering
    if (options?.filter) {
      users = this.applyFilter(users, options.filter);
    }

    // Apply sorting
    if (options?.sort) {
      users = this.applySort(users, options.sort);
    }

    // Apply pagination
    const pagination = options?.pagination || { page: 1, limit: 10 };
    const startIndex = (pagination.page - 1) * pagination.limit;
    const endIndex = startIndex + pagination.limit;
    const paginatedUsers = users.slice(startIndex, endIndex);

    return {
      data: paginatedUsers,
      pagination: {
        page: pagination.page,
        limit: pagination.limit,
        total: users.length,
        totalPages: Math.ceil(users.length / pagination.limit),
        hasNext: endIndex < users.length,
        hasPrev: pagination.page > 1
      }
    };
  }

  async create(userData: Omit<User, 'id' | keyof Timestamped>): Promise<User> {
    const userResult = User.create({
      firstName: userData.firstName,
      lastName: userData.lastName,
      email: userData.email.toString(),
      role: userData.role,
      preferences: userData.preferences,
      profile: userData.profile
    });

    if (userResult.isErr()) {
      throw userResult.unwrapErr();
    }

    const user = userResult.unwrap();
    this.users.set(user.id, user);
    return user;
  }

  async update(id: UserId, updates: DeepPartial<Omit<User, 'id' | 'createdAt'>>): Promise<User> {
    const user = await this.findById(id);
    if (!user) {
      throw new NotFoundError('User', id);
    }

    // Apply updates (simplified for example)
    if (updates.firstName) user.firstName = updates.firstName;
    if (updates.lastName) user.lastName = updates.lastName;
    if (updates.role) user.role = updates.role;
    if (updates.isActive !== undefined) user.isActive = updates.isActive;

    this.users.set(id, user);
    return user;
  }

  async delete(id: UserId): Promise<boolean> {
    return this.users.delete(id);
  }

  async count(filter?: FilterCriteria<User>): Promise<number> {
    if (!filter) return this.users.size;
    
    const users = Array.from(this.users.values());
    return this.applyFilter(users, filter).length;
  }

  async findByEmail(email: string): Promise<User | null> {
    for (const user of this.users.values()) {
      if (user.email.toString() === email) {
        return user;
      }
    }
    return null;
  }

  async findByRole(role: UserRole): Promise<User[]> {
    return Array.from(this.users.values()).filter(user => user.role === role);
  }

  private applyFilter(users: User[], filter: FilterCriteria<User>): User[] {
    return users.filter(user => {
      for (const [key, criteria] of Object.entries(filter)) {
        const value = (user as any)[key];
        
        if (typeof criteria === 'object' && criteria !== null) {
          // Handle complex filter criteria
          if ('equals' in criteria && value !== criteria.equals) return false;
          if ('contains' in criteria && typeof value === 'string' && !value.includes(criteria.contains!)) return false;
          // Add more filter logic as needed
        } else if (value !== criteria) {
          return false;
        }
      }
      return true;
    });
  }

  private applySort(users: User[], sort: SortCriteria<User>): User[] {
    return users.sort((a, b) => {
      for (const [key, direction] of Object.entries(sort)) {
        const aValue = (a as any)[key];
        const bValue = (b as any)[key];
        
        let comparison = 0;
        if (aValue < bValue) comparison = -1;
        else if (aValue > bValue) comparison = 1;
        
        if (comparison !== 0) {
          return direction === 'desc' ? -comparison : comparison;
        }
      }
      return 0;
    });
  }
}

// ============ Service Layer with Functional Composition ============

// Higher-order functions for validation and transformation
type Validator<T> = (value: T) => Result<T, ValidationError>;
type Transformer<T, U> = (value: T) => U;

const compose = <T, U, V>(f: (b: U) => V, g: (a: T) => U) => (a: T): V => f(g(a));

const pipe = <T>(...fns: Array<(arg: T) => T>) => (value: T): T =>
  fns.reduce((acc, fn) => fn(acc), value);

// Validation combinators
const required = <T>(fieldName: string): Validator<T | undefined> => (value) => {
  if (value === undefined || value === null) {
    return Result.failure(new ValidationError(`${fieldName} is required`, fieldName));
  }
  return Result.success(value);
};

const minLength = (min: number, fieldName: string): Validator<string> => (value) => {
  if (value.length < min) {
    return Result.failure(new ValidationError(`${fieldName} must be at least ${min} characters`, fieldName));
  }
  return Result.success(value);
};

const maxLength = (max: number, fieldName: string): Validator<string> => (value) => {
  if (value.length > max) {
    return Result.failure(new ValidationError(`${fieldName} cannot exceed ${max} characters`, fieldName));
  }
  return Result.success(value);
};

const email = (fieldName: string): Validator<string> => (value) => {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  if (!emailRegex.test(value)) {
    return Result.failure(new ValidationError(`${fieldName} must be a valid email`, fieldName));
  }
  return Result.success(value);
};

// Combine validators
const validateWith = <T>(...validators: Validator<T>[]): Validator<T> => (value) => {
  for (const validator of validators) {
    const result = validator(value);
    if (result.isErr()) {
      return result;
    }
  }
  return Result.success(value);
};

// Service classes with rich behavior
class UserService {
  constructor(
    private userRepository: Repository<User, UserId>,
    private eventEmitter: EventEmitter
  ) {}

  async createUser(userData: {
    firstName: string;
    lastName: string;
    email: string;
    role?: UserRole;
    preferences?: Partial<UserPreferences>;
    profile?: Partial<UserProfile>;
  }): Promise<Result<User, ValidationError | DomainError>> {
    try {
      // Validate input
      const firstNameResult = validateWith(
        required('firstName'),
        minLength(1, 'firstName'),
        maxLength(50, 'firstName')
      )(userData.firstName);

      if (firstNameResult.isErr()) return firstNameResult;

      const lastNameResult = validateWith(
        required('lastName'),
        minLength(1, 'lastName'),
        maxLength(50, 'lastName')
      )(userData.lastName);

      if (lastNameResult.isErr()) return lastNameResult;

      const emailResult = validateWith(
        required('email'),
        email('email')
      )(userData.email);

      if (emailResult.isErr()) return emailResult;

      // Check if user already exists
      const existingUser = await this.userRepository.findByEmail?.(userData.email);
      if (existingUser) {
        return Result.failure(new ValidationError('Email already exists', 'email'));
      }

      // Create user
      const userResult = User.create(userData);
      if (userResult.isErr()) {
        return userResult;
      }

      const user = userResult.unwrap();
      const savedUser = await this.userRepository.create(user);

      // Emit event
      this.eventEmitter.emit('user:created', { user: savedUser });

      return Result.success(savedUser);
    } catch (error) {
      return Result.failure(error as DomainError);
    }
  }

  async getUserById(id: UserId): Promise<Result<User, NotFoundError>> {
    const user = await this.userRepository.findById(id);
    if (!user) {
      return Result.failure(new NotFoundError('User', id));
    }
    return Result.success(user);
  }

  async updateUser(
    id: UserId,
    updates: DeepPartial<Pick<User, 'firstName' | 'lastName' | 'profile'>>
  ): Promise<Result<User, ValidationError | NotFoundError>> {
    try {
      const userResult = await this.getUserById(id);
      if (userResult.isErr()) {
        return userResult;
      }

      const updatedUser = await this.userRepository.update(id, updates);
      
      this.eventEmitter.emit('user:updated', { user: updatedUser });
      
      return Result.success(updatedUser);
    } catch (error) {
      return Result.failure(error as ValidationError);
    }
  }

  async searchUsers(criteria: {
    query?: string;
    role?: UserRole;
    isActive?: boolean;
    page?: number;
    limit?: number;
  }): Promise<PaginatedResult<User>> {
    const filter: FilterCriteria<User> = {};
    
    if (criteria.role) {
      filter.role = criteria.role;
    }
    
    if (criteria.isActive !== undefined) {
      filter.isActive = criteria.isActive;
    }

    return this.userRepository.findAll({
      filter,
      pagination: {
        page: criteria.page || 1,
        limit: criteria.limit || 10
      }
    });
  }

  async authenticateUser(email: string, password: string): Promise<Result<User, UnauthorizedError>> {
    // Simplified authentication logic
    const user = await this.userRepository.findByEmail?.(email);
    
    if (!user || !user.isActive) {
      return Result.failure(new UnauthorizedError('Invalid credentials'));
    }

    // In a real implementation, you would verify the password hash
    // const isPasswordValid = await this.passwordService.verify(password, user.passwordHash);
    const isPasswordValid = true; // Simplified for example

    if (!isPasswordValid) {
      return Result.failure(new UnauthorizedError('Invalid credentials'));
    }

    user.recordLogin();
    await this.userRepository.update(user.id, { lastLoginAt: user.lastLoginAt });

    this.eventEmitter.emit('user:login', { user });

    return Result.success(user);
  }
}

class TaskService {
  constructor(
    private taskRepository: Repository<Task, number>,
    private userRepository: Repository<User, UserId>,
    private eventEmitter: EventEmitter
  ) {}

  async createTask(taskData: {
    title: string;
    description: string;
    priority: Priority;
    assigneeId: UserId;
    createdBy: UserId;
    dueDate?: Date;
    estimatedHours?: number;
    tags?: string[];
  }): Promise<Result<Task, ValidationError | NotFoundError>> {
    try {
      // Validate that assignee exists
      const assigneeResult = await this.userRepository.findById(taskData.assigneeId);
      if (!assigneeResult) {
        return Result.failure(new NotFoundError('User', taskData.assigneeId));
      }

      // Validate that creator exists
      const creatorResult = await this.userRepository.findById(taskData.createdBy);
      if (!creatorResult) {
        return Result.failure(new NotFoundError('User', taskData.createdBy));
      }

      // Create task
      const taskResult = Task.create(taskData);
      if (taskResult.isErr()) {
        return taskResult;
      }

      const task = taskResult.unwrap();
      const savedTask = await this.taskRepository.create(task);

      this.eventEmitter.emit('task:created', { task: savedTask, assignee: assigneeResult });

      return Result.success(savedTask);
    } catch (error) {
      return Result.failure(error as ValidationError);
    }
  }

  async updateTaskStatus(
    taskId: number,
    newStatus: TaskStatus,
    userId: UserId
  ): Promise<Result<Task, ValidationError | NotFoundError | UnauthorizedError>> {
    try {
      const task = await this.taskRepository.findById(taskId);
      if (!task) {
        return Result.failure(new NotFoundError('Task', taskId.toString()));
      }

      // Check permissions
      if (task.assigneeId !== userId && task.createdBy !== userId) {
        const user = await this.userRepository.findById(userId);
        if (!user || !user.hasPermission('write:all')) {
          return Result.failure(new UnauthorizedError('Insufficient permissions'));
        }
      }

      const statusResult = task.changeStatus(newStatus);
      if (statusResult.isErr()) {
        return statusResult;
      }

      const updatedTask = await this.taskRepository.update(taskId, { status: newStatus });
      
      this.eventEmitter.emit('task:status_changed', { 
        task: updatedTask, 
        oldStatus: task.status,
        newStatus,
        changedBy: userId
      });

      return Result.success(updatedTask);
    } catch (error) {
      return Result.failure(error as ValidationError);
    }
  }

  async getTaskStatistics(): Promise<{
    total: number;
    byStatus: Record<TaskStatus, number>;
    byPriority: Record<Priority, number>;
    overdue: number;
    completionRate: number;
  }> {
    const allTasks = await this.taskRepository.findAll({ pagination: { page: 1, limit: 1000 } });
    const tasks = allTasks.data;

    const byStatus = tasks.reduce((acc, task) => {
      acc[task.status] = (acc[task.status] || 0) + 1;
      return acc;
    }, {} as Record<TaskStatus, number>);

    const byPriority = tasks.reduce((acc, task) => {
      acc[task.priority] = (acc[task.priority] || 0) + 1;
      return acc;
    }, {} as Record<Priority, number>);

    const overdue = tasks.filter(task => task.isOverdue).length;
    const completed = byStatus[TaskStatus.COMPLETED] || 0;
    const completionRate = tasks.length > 0 ? (completed / tasks.length) * 100 : 0;

    return {
      total: tasks.length,
      byStatus,
      byPriority,
      overdue,
      completionRate
    };
  }
}

// ============ Advanced Async Patterns ============

// Promise utilities
class PromiseUtils {
  static async retry<T>(
    fn: () => Promise<T>,
    options: {
      retries: number;
      delay: number;
      backoff?: 'linear' | 'exponential';
    }
  ): Promise<T> {
    let lastError: Error;
    
    for (let i = 0; i <= options.retries; i++) {
      try {
        return await fn();
      } catch (error) {
        lastError = error as Error;
        
        if (i < options.retries) {
          const delay = options.backoff === 'exponential' 
            ? options.delay * Math.pow(2, i)
            : options.delay;
          
          await this.delay(delay);
        }
      }
    }
    
    throw lastError;
  }

  static async timeout<T>(promise: Promise<T>, ms: number): Promise<T> {
    const timeoutPromise = new Promise<never>((_, reject) => {
      setTimeout(() => reject(new Error(`Operation timed out after ${ms}ms`)), ms);
    });

    return Promise.race([promise, timeoutPromise]);
  }

  static async batch<T, U>(
    items: T[],
    processor: (item: T) => Promise<U>,
    batchSize: number = 10
  ): Promise<U[]> {
    const results: U[] = [];
    
    for (let i = 0; i < items.length; i += batchSize) {
      const batch = items.slice(i, i + batchSize);
      const batchResults = await Promise.all(batch.map(processor));
      results.push(...batchResults);
    }
    
    return results;
  }

  static async sequential<T, U>(
    items: T[],
    processor: (item: T, index: number) => Promise<U>
  ): Promise<U[]> {
    const results: U[] = [];
    
    for (let i = 0; i < items.length; i++) {
      const result = await processor(items[i], i);
      results.push(result);
    }
    
    return results;
  }

  private static delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

// Event-driven architecture
class EventBus extends EventEmitter {
  private handlers = new Map<string, Array<(data: any) => Promise<void>>>();

  onAsync<T = any>(event: string, handler: (data: T) => Promise<void>): void {
    if (!this.handlers.has(event)) {
      this.handlers.set(event, []);
    }
    this.handlers.get(event)!.push(handler);
  }

  async emitAsync<T = any>(event: string, data: T): Promise<void> {
    const handlers = this.handlers.get(event) || [];
    await Promise.all(handlers.map(handler => handler(data)));
  }

  async emitSequential<T = any>(event: string, data: T): Promise<void> {
    const handlers = this.handlers.get(event) || [];
    for (const handler of handlers) {
      await handler(data);
    }
  }
}

// ============ Demo Application ============

async function runDemo(): Promise<void> {
  console.log('=== TypeScript Comprehensive Examples Demo ===\n');

  // Setup
  const eventBus = new EventBus();
  const userRepository = new InMemoryUserRepository();
  const taskRepository = new InMemoryTaskRepository(); // Assume similar implementation
  
  const userService = new UserService(userRepository, eventBus);
  const taskService = new TaskService(taskRepository, userRepository, eventBus);

  // Event handlers
  eventBus.onAsync('user:created', async (data) => {
    console.log(`📧 Welcome email sent to ${data.user.email}`);
  });

  eventBus.onAsync('task:created', async (data) => {
    console.log(`📋 Task assigned notification sent to ${data.assignee.email}`);
  });

  try {
    // Create users
    console.log('Creating users...');
    
    const adminResult = await userService.createUser({
      firstName: 'Admin',
      lastName: 'User',
      email: 'admin@example.com',
      role: UserRole.ADMIN
    });

    if (adminResult.isErr()) {
      console.error('Failed to create admin:', adminResult.unwrapErr().message);
      return;
    }

    const user1Result = await userService.createUser({
      firstName: 'Alice',
      lastName: 'Johnson',
      email: 'alice@example.com',
      preferences: {
        theme: 'dark',
        language: 'en',
        timezone: 'UTC',
        notifications: { email: true, push: true, sms: false }
      }
    });

    const user2Result = await userService.createUser({
      firstName: 'Bob',
      lastName: 'Smith',
      email: 'bob@example.com'
    });

    if (user1Result.isErr() || user2Result.isErr()) {
      console.error('Failed to create users');
      return;
    }

    const admin = adminResult.unwrap();
    const alice = user1Result.unwrap();
    const bob = user2Result.unwrap();

    console.log(`✅ Created ${[admin, alice, bob].length} users\n`);

    // Create tasks
    console.log('Creating tasks...');
    
    const task1Result = await taskService.createTask({
      title: 'Implement user authentication',
      description: 'Add secure login system with JWT tokens',
      priority: Priority.HIGH,
      assigneeId: alice.id,
      createdBy: admin.id,
      dueDate: new Date(Date.now() + 7 * 24 * 60 * 60 * 1000), // 7 days from now
      estimatedHours: 16
    });

    const task2Result = await taskService.createTask({
      title: 'Design database schema',
      description: 'Create comprehensive database design',
      priority: Priority.MEDIUM,
      assigneeId: bob.id,
      createdBy: admin.id,
      estimatedHours: 8
    });

    if (task1Result.isErr() || task2Result.isErr()) {
      console.error('Failed to create tasks');
      return;
    }

    const task1 = task1Result.unwrap();
    const task2 = task2Result.unwrap();

    console.log('✅ Created 2 tasks\n');

    // Update task statuses
    console.log('Updating task statuses...');
    
    await taskService.updateTaskStatus(task1.id, TaskStatus.IN_PROGRESS, alice.id);
    await taskService.updateTaskStatus(task2.id, TaskStatus.COMPLETED, bob.id);

    // Get statistics
    console.log('Task Statistics:');
    const stats = await taskService.getTaskStatistics();
    console.log('📊 Statistics:', JSON.stringify(stats, null, 2));

    // Test advanced features
    console.log('\n=== Advanced TypeScript Features Demo ===');

    // Test Result type
    const emailResult = EmailAddress.create('test@example.com');
    emailResult.match(
      email => console.log(`✅ Valid email: ${email.getValue()}`),
      error => console.log(`❌ Invalid email: ${error.message}`)
    );

    // Test Money value object
    const money1Result = Money.create(100.50, 'USD');
    const money2Result = Money.create(50.25, 'USD');

    if (money1Result.isOk() && money2Result.isOk()) {
      const money1 = money1Result.unwrap();
      const money2 = money2Result.unwrap();
      
      const sumResult = money1.add(money2);
      sumResult.match(
        sum => console.log(`💰 Sum: ${sum.toString()}`),
        error => console.log(`❌ Error: ${error.message}`)
      );
    }

    // Test async utilities
    console.log('\n📡 Testing async utilities...');
    
    const retryResult = await PromiseUtils.retry(
      async () => {
        if (Math.random() > 0.7) {
          return 'Success!';
        }
        throw new Error('Random failure');
      },
      { retries: 3, delay: 100 }
    );
    
    console.log(`🔄 Retry result: ${retryResult}`);

    // Test batch processing
    const items = Array.from({ length: 15 }, (_, i) => i + 1);
    const processed = await PromiseUtils.batch(
      items,
      async (item) => {
        await PromiseUtils.delay(10);
        return item * 2;
      },
      5
    );
    
    console.log(`🔧 Batch processed: [${processed.slice(0, 5).join(', ')}...]`);

    // Test search functionality
    console.log('\n🔍 Testing search...');
    const searchResults = await userService.searchUsers({
      query: 'alice',
      isActive: true,
      page: 1,
      limit: 10
    });
    
    console.log(`📋 Found ${searchResults.data.length} users matching 'alice'`);

  } catch (error) {
    console.error('❌ Demo error:', error);
  }

  console.log('\n=== TypeScript Features Demonstrated ===');
  console.log('✨ Advanced type system with utility types');
  console.log('🏗️ Generic constraints and conditional types');
  console.log('🏷️ Template literal types and branded types');
  console.log('🎯 Discriminated unions and pattern matching');
  console.log('📦 Value objects with validation');
  console.log('🛡️ Result type for error handling');
  console.log('🔧 Higher-order functions and composition');
  console.log('⚡ Async/await with advanced patterns');
  console.log('📡 Event-driven architecture');
  console.log('🏛️ Repository and service patterns');
  console.log('🧪 Domain-driven design principles');
  console.log('🎨 Functional programming concepts');
  console.log('🔒 Type safety and null safety');
  console.log('🚀 Modern ES2022+ features');
}

// Placeholder for missing imports/implementations
class InMemoryTaskRepository implements Repository<Task, number> {
  private tasks = new Map<number, Task>();

  async findById(id: number): Promise<Task | null> {
    return this.tasks.get(id) || null;
  }

  async findAll(options?: QueryOptions<Task>): Promise<PaginatedResult<Task>> {
    const tasks = Array.from(this.tasks.values());
    return {
      data: tasks,
      pagination: {
        page: 1,
        limit: tasks.length,
        total: tasks.length,
        totalPages: 1,
        hasNext: false,
        hasPrev: false
      }
    };
  }

  async create(taskData: Omit<Task, 'id' | keyof Timestamped>): Promise<Task> {
    const task = taskData as Task;
    this.tasks.set(task.id, task);
    return task;
  }

  async update(id: number, updates: DeepPartial<Omit<Task, 'id' | 'createdAt'>>): Promise<Task> {
    const task = this.tasks.get(id);
    if (!task) throw new NotFoundError('Task', id.toString());
    
    Object.assign(task, updates);
    return task;
  }

  async delete(id: number): Promise<boolean> {
    return this.tasks.delete(id);
  }

  async count(): Promise<number> {
    return this.tasks.size;
  }
}

// Run demo if this file is executed directly
if (require.main === module) {
  runDemo().catch(console.error);
}

export {
  User,
  Task,
  UserService,
  TaskService,
  Result,
  EmailAddress,
  Money,
  PromiseUtils,
  EventBus,
  ValidationError,
  BusinessRuleError,
  NotFoundError,
  UnauthorizedError
};