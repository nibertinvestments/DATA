// Advanced Object-Oriented Programming concepts in Java

import java.util.*;
import java.util.concurrent.*;
import java.util.stream.*;
import java.lang.reflect.*;
import java.io.*;
import java.nio.file.*;

/**
 * Comprehensive Java OOP examples covering advanced concepts
 * for ML/AI training datasets
 */

// Abstract base class demonstrating inheritance and polymorphism
abstract class Shape {
    protected String color;
    protected double borderWidth;
    protected static int shapeCounter = 0;
    
    public Shape(String color, double borderWidth) {
        this.color = color;
        this.borderWidth = borderWidth;
        shapeCounter++;
    }
    
    // Abstract methods - must be implemented by subclasses
    public abstract double calculateArea();
    public abstract double calculatePerimeter();
    public abstract String getShapeType();
    
    // Concrete method with default implementation
    public String getDescription() {
        return String.format("%s %s with area %.2f and perimeter %.2f", 
            color, getShapeType(), calculateArea(), calculatePerimeter());
    }
    
    // Static method
    public static int getShapeCount() {
        return shapeCounter;
    }
    
    // Getter and setter methods
    public String getColor() { return color; }
    public void setColor(String color) { this.color = color; }
    public double getBorderWidth() { return borderWidth; }
    public void setBorderWidth(double borderWidth) { this.borderWidth = borderWidth; }
}

// Interface for drawable objects
interface Drawable {
    void draw(Graphics graphics);
    boolean isVisible();
    void setVisible(boolean visible);
}

// Interface for movable objects
interface Movable {
    void move(double deltaX, double deltaY);
    Point getPosition();
    void setPosition(Point position);
}

// Interface for scalable objects
interface Scalable {
    void scale(double factor);
    double getScale();
}

// Point class for position handling
class Point {
    private double x, y;
    
    public Point(double x, double y) {
        this.x = x;
        this.y = y;
    }
    
    public double getX() { return x; }
    public double getY() { return y; }
    public void setX(double x) { this.x = x; }
    public void setY(double y) { this.y = y; }
    
    public double distanceTo(Point other) {
        return Math.sqrt(Math.pow(this.x - other.x, 2) + Math.pow(this.y - other.y, 2));
    }
    
    @Override
    public String toString() {
        return String.format("Point(%.2f, %.2f)", x, y);
    }
    
    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (obj == null || getClass() != obj.getClass()) return false;
        Point point = (Point) obj;
        return Double.compare(point.x, x) == 0 && Double.compare(point.y, y) == 0;
    }
    
    @Override
    public int hashCode() {
        return Objects.hash(x, y);
    }
}

// Graphics class (simplified for demonstration)
class Graphics {
    private PrintWriter output;
    
    public Graphics(PrintWriter output) {
        this.output = output;
    }
    
    public void drawLine(Point start, Point end) {
        output.printf("Drawing line from %s to %s%n", start, end);
    }
    
    public void drawCircle(Point center, double radius) {
        output.printf("Drawing circle at %s with radius %.2f%n", center, radius);
    }
    
    public void drawRectangle(Point topLeft, double width, double height) {
        output.printf("Drawing rectangle at %s with dimensions %.2f x %.2f%n", 
            topLeft, width, height);
    }
}

// Concrete implementation of Circle
class Circle extends Shape implements Drawable, Movable, Scalable {
    private double radius;
    private Point center;
    private boolean visible;
    private double scale;
    
    public Circle(String color, double borderWidth, double radius, Point center) {
        super(color, borderWidth);
        this.radius = radius;
        this.center = center;
        this.visible = true;
        this.scale = 1.0;
    }
    
    @Override
    public double calculateArea() {
        return Math.PI * radius * radius * scale * scale;
    }
    
    @Override
    public double calculatePerimeter() {
        return 2 * Math.PI * radius * scale;
    }
    
    @Override
    public String getShapeType() {
        return "Circle";
    }
    
    @Override
    public void draw(Graphics graphics) {
        if (visible) {
            graphics.drawCircle(center, radius * scale);
        }
    }
    
    @Override
    public boolean isVisible() {
        return visible;
    }
    
    @Override
    public void setVisible(boolean visible) {
        this.visible = visible;
    }
    
    @Override
    public void move(double deltaX, double deltaY) {
        center.setX(center.getX() + deltaX);
        center.setY(center.getY() + deltaY);
    }
    
    @Override
    public Point getPosition() {
        return new Point(center.getX(), center.getY());
    }
    
    @Override
    public void setPosition(Point position) {
        this.center = new Point(position.getX(), position.getY());
    }
    
    @Override
    public void scale(double factor) {
        this.scale *= factor;
    }
    
    @Override
    public double getScale() {
        return scale;
    }
    
    // Circle-specific methods
    public double getRadius() { return radius; }
    public void setRadius(double radius) { this.radius = radius; }
    public Point getCenter() { return center; }
}

// Concrete implementation of Rectangle
class Rectangle extends Shape implements Drawable, Movable, Scalable {
    private double width, height;
    private Point topLeft;
    private boolean visible;
    private double scale;
    
    public Rectangle(String color, double borderWidth, double width, double height, Point topLeft) {
        super(color, borderWidth);
        this.width = width;
        this.height = height;
        this.topLeft = topLeft;
        this.visible = true;
        this.scale = 1.0;
    }
    
    @Override
    public double calculateArea() {
        return width * height * scale * scale;
    }
    
    @Override
    public double calculatePerimeter() {
        return 2 * (width + height) * scale;
    }
    
    @Override
    public String getShapeType() {
        return "Rectangle";
    }
    
    @Override
    public void draw(Graphics graphics) {
        if (visible) {
            graphics.drawRectangle(topLeft, width * scale, height * scale);
        }
    }
    
    @Override
    public boolean isVisible() {
        return visible;
    }
    
    @Override
    public void setVisible(boolean visible) {
        this.visible = visible;
    }
    
    @Override
    public void move(double deltaX, double deltaY) {
        topLeft.setX(topLeft.getX() + deltaX);
        topLeft.setY(topLeft.getY() + deltaY);
    }
    
    @Override
    public Point getPosition() {
        return new Point(topLeft.getX(), topLeft.getY());
    }
    
    @Override
    public void setPosition(Point position) {
        this.topLeft = new Point(position.getX(), position.getY());
    }
    
    @Override
    public void scale(double factor) {
        this.scale *= factor;
    }
    
    @Override
    public double getScale() {
        return scale;
    }
    
    // Rectangle-specific methods
    public double getWidth() { return width; }
    public double getHeight() { return height; }
    public void setWidth(double width) { this.width = width; }
    public void setHeight(double height) { this.height = height; }
}

// Generic class for shape collections
class ShapeCollection<T extends Shape> implements Iterable<T> {
    private List<T> shapes;
    private String collectionName;
    
    public ShapeCollection(String collectionName) {
        this.shapes = new ArrayList<>();
        this.collectionName = collectionName;
    }
    
    public void addShape(T shape) {
        shapes.add(shape);
    }
    
    public boolean removeShape(T shape) {
        return shapes.remove(shape);
    }
    
    public T getShape(int index) {
        return shapes.get(index);
    }
    
    public int size() {
        return shapes.size();
    }
    
    public double getTotalArea() {
        return shapes.stream()
                    .mapToDouble(Shape::calculateArea)
                    .sum();
    }
    
    public double getTotalPerimeter() {
        return shapes.stream()
                    .mapToDouble(Shape::calculatePerimeter)
                    .sum();
    }
    
    public List<T> getShapesByColor(String color) {
        return shapes.stream()
                    .filter(shape -> shape.getColor().equalsIgnoreCase(color))
                    .collect(Collectors.toList());
    }
    
    public Optional<T> getLargestShape() {
        return shapes.stream()
                    .max(Comparator.comparingDouble(Shape::calculateArea));
    }
    
    public Map<String, List<T>> groupByColor() {
        return shapes.stream()
                    .collect(Collectors.groupingBy(Shape::getColor));
    }
    
    @Override
    public Iterator<T> iterator() {
        return shapes.iterator();
    }
    
    public String getCollectionName() {
        return collectionName;
    }
}

// Design Pattern Examples

// Factory Pattern
abstract class ShapeFactory {
    public abstract Shape createShape(String type, String color, double borderWidth, double... params);
    
    public static ShapeFactory getInstance() {
        return new ConcreteShapeFactory();
    }
}

class ConcreteShapeFactory extends ShapeFactory {
    @Override
    public Shape createShape(String type, String color, double borderWidth, double... params) {
        switch (type.toLowerCase()) {
            case "circle":
                if (params.length >= 3) {
                    return new Circle(color, borderWidth, params[0], 
                        new Point(params[1], params[2]));
                }
                break;
            case "rectangle":
                if (params.length >= 4) {
                    return new Rectangle(color, borderWidth, params[0], params[1], 
                        new Point(params[2], params[3]));
                }
                break;
        }
        throw new IllegalArgumentException("Invalid shape type or parameters");
    }
}

// Observer Pattern
interface ShapeObserver {
    void onShapeChanged(Shape shape, String changeType);
}

class ShapeManager {
    private List<Shape> shapes;
    private List<ShapeObserver> observers;
    
    public ShapeManager() {
        this.shapes = new ArrayList<>();
        this.observers = new ArrayList<>();
    }
    
    public void addObserver(ShapeObserver observer) {
        observers.add(observer);
    }
    
    public void removeObserver(ShapeObserver observer) {
        observers.remove(observer);
    }
    
    public void addShape(Shape shape) {
        shapes.add(shape);
        notifyObservers(shape, "ADDED");
    }
    
    public void removeShape(Shape shape) {
        if (shapes.remove(shape)) {
            notifyObservers(shape, "REMOVED");
        }
    }
    
    public void updateShape(Shape shape) {
        notifyObservers(shape, "UPDATED");
    }
    
    private void notifyObservers(Shape shape, String changeType) {
        for (ShapeObserver observer : observers) {
            observer.onShapeChanged(shape, changeType);
        }
    }
    
    public List<Shape> getShapes() {
        return new ArrayList<>(shapes);
    }
}

// Observer implementation
class ShapeLogger implements ShapeObserver {
    private PrintWriter logWriter;
    
    public ShapeLogger(PrintWriter logWriter) {
        this.logWriter = logWriter;
    }
    
    @Override
    public void onShapeChanged(Shape shape, String changeType) {
        logWriter.printf("[%s] Shape %s: %s%n", 
            new Date(), changeType, shape.getDescription());
    }
}

// Decorator Pattern
abstract class ShapeDecorator extends Shape {
    protected Shape decoratedShape;
    
    public ShapeDecorator(Shape decoratedShape) {
        super(decoratedShape.getColor(), decoratedShape.getBorderWidth());
        this.decoratedShape = decoratedShape;
    }
    
    @Override
    public double calculateArea() {
        return decoratedShape.calculateArea();
    }
    
    @Override
    public double calculatePerimeter() {
        return decoratedShape.calculatePerimeter();
    }
    
    @Override
    public String getShapeType() {
        return decoratedShape.getShapeType();
    }
}

class BorderDecorator extends ShapeDecorator {
    private String borderStyle;
    private double additionalBorderWidth;
    
    public BorderDecorator(Shape decoratedShape, String borderStyle, double additionalBorderWidth) {
        super(decoratedShape);
        this.borderStyle = borderStyle;
        this.additionalBorderWidth = additionalBorderWidth;
    }
    
    @Override
    public String getDescription() {
        return String.format("%s with %s border (width: %.2f)", 
            decoratedShape.getDescription(), borderStyle, 
            getBorderWidth() + additionalBorderWidth);
    }
    
    @Override
    public double getBorderWidth() {
        return super.getBorderWidth() + additionalBorderWidth;
    }
}

// Strategy Pattern
interface AreaCalculationStrategy {
    double calculateArea(Shape shape);
}

class PreciseAreaCalculation implements AreaCalculationStrategy {
    @Override
    public double calculateArea(Shape shape) {
        return shape.calculateArea();
    }
}

class ApproximateAreaCalculation implements AreaCalculationStrategy {
    @Override
    public double calculateArea(Shape shape) {
        // Simplified approximation
        return Math.round(shape.calculateArea() / 10.0) * 10.0;
    }
}

class AreaCalculator {
    private AreaCalculationStrategy strategy;
    
    public AreaCalculator(AreaCalculationStrategy strategy) {
        this.strategy = strategy;
    }
    
    public void setStrategy(AreaCalculationStrategy strategy) {
        this.strategy = strategy;
    }
    
    public double calculateArea(Shape shape) {
        return strategy.calculateArea(shape);
    }
}

// Exception handling
class ShapeException extends Exception {
    public ShapeException(String message) {
        super(message);
    }
}

class InvalidShapeParameterException extends ShapeException {
    public InvalidShapeParameterException(String parameter, double value) {
        super(String.format("Invalid parameter %s: %.2f", parameter, value));
    }
}

// Utility class with static methods
final class ShapeUtils {
    private ShapeUtils() {} // Prevent instantiation
    
    public static double calculateDistance(Shape shape1, Shape shape2) {
        if (shape1 instanceof Movable && shape2 instanceof Movable) {
            Point pos1 = ((Movable) shape1).getPosition();
            Point pos2 = ((Movable) shape2).getPosition();
            return pos1.distanceTo(pos2);
        }
        throw new IllegalArgumentException("Shapes must be movable to calculate distance");
    }
    
    public static boolean shapesOverlap(Circle circle1, Circle circle2) {
        double distance = calculateDistance(circle1, circle2);
        double radiusSum = circle1.getRadius() * circle1.getScale() + 
                          circle2.getRadius() * circle2.getScale();
        return distance < radiusSum;
    }
    
    public static List<Shape> sortShapesByArea(List<Shape> shapes) {
        return shapes.stream()
                    .sorted(Comparator.comparingDouble(Shape::calculateArea))
                    .collect(Collectors.toList());
    }
    
    public static Map<String, DoubleSummaryStatistics> getAreaStatisticsByType(List<Shape> shapes) {
        return shapes.stream()
                    .collect(Collectors.groupingBy(
                        Shape::getShapeType,
                        Collectors.summarizingDouble(Shape::calculateArea)
                    ));
    }
}

// Reflection example
class ShapeInspector {
    public static void inspectShape(Shape shape) {
        Class<?> clazz = shape.getClass();
        System.out.printf("Inspecting shape: %s%n", clazz.getSimpleName());
        
        // Get all methods
        Method[] methods = clazz.getMethods();
        System.out.println("Methods:");
        for (Method method : methods) {
            if (method.getDeclaringClass() != Object.class) {
                System.out.printf("  %s%n", method.getName());
            }
        }
        
        // Get all fields
        Field[] fields = clazz.getDeclaredFields();
        System.out.println("Fields:");
        for (Field field : fields) {
            System.out.printf("  %s %s%n", field.getType().getSimpleName(), field.getName());
        }
        
        // Get interfaces
        Class<?>[] interfaces = clazz.getInterfaces();
        System.out.println("Interfaces:");
        for (Class<?> iface : interfaces) {
            System.out.printf("  %s%n", iface.getSimpleName());
        }
    }
}

// Main demonstration class
public class AdvancedOOPDemo {
    
    public static void main(String[] args) {
        System.out.println("=== Advanced Java OOP Demonstration ===\n");
        
        try (PrintWriter output = new PrintWriter(System.out, true)) {
            Graphics graphics = new Graphics(output);
            
            // Create shapes using factory
            ShapeFactory factory = ShapeFactory.getInstance();
            
            Circle circle = (Circle) factory.createShape("circle", "red", 2.0, 5.0, 10.0, 15.0);
            Rectangle rectangle = (Rectangle) factory.createShape("rectangle", "blue", 1.5, 8.0, 6.0, 5.0, 10.0);
            
            System.out.println("=== Shape Creation ===");
            System.out.println(circle.getDescription());
            System.out.println(rectangle.getDescription());
            System.out.println("Total shapes created: " + Shape.getShapeCount());
            
            // Test polymorphism
            System.out.println("\n=== Polymorphism ===");
            List<Shape> shapes = Arrays.asList(circle, rectangle);
            for (Shape shape : shapes) {
                System.out.printf("Shape: %s, Area: %.2f%n", 
                    shape.getShapeType(), shape.calculateArea());
            }
            
            // Test interfaces
            System.out.println("\n=== Interface Implementation ===");
            List<Drawable> drawables = Arrays.asList(circle, rectangle);
            for (Drawable drawable : drawables) {
                drawable.draw(graphics);
            }
            
            // Test movement
            System.out.println("\n=== Movement ===");
            System.out.println("Original position: " + circle.getPosition());
            circle.move(5.0, -3.0);
            System.out.println("New position: " + circle.getPosition());
            
            // Test scaling
            System.out.println("\n=== Scaling ===");
            System.out.printf("Original area: %.2f%n", circle.calculateArea());
            circle.scale(1.5);
            System.out.printf("Scaled area: %.2f%n", circle.calculateArea());
            
            // Test collections with generics
            System.out.println("\n=== Generic Collections ===");
            ShapeCollection<Circle> circles = new ShapeCollection<>("Circles");
            circles.addShape(circle);
            circles.addShape(new Circle("green", 1.0, 3.0, new Point(20, 25)));
            
            System.out.printf("Circle collection total area: %.2f%n", circles.getTotalArea());
            
            // Test streams and functional programming
            System.out.println("\n=== Stream Operations ===");
            ShapeCollection<Shape> allShapes = new ShapeCollection<>("All Shapes");
            allShapes.addShape(circle);
            allShapes.addShape(rectangle);
            allShapes.addShape(new Circle("yellow", 2.0, 4.0, new Point(0, 0)));
            
            Map<String, List<Shape>> groupedByColor = allShapes.groupByColor();
            groupedByColor.forEach((color, shapeList) -> {
                System.out.printf("Color %s: %d shapes%n", color, shapeList.size());
            });
            
            Optional<Shape> largestShape = allShapes.getLargestShape();
            largestShape.ifPresent(shape -> 
                System.out.printf("Largest shape: %s%n", shape.getDescription()));
            
            // Test Observer pattern
            System.out.println("\n=== Observer Pattern ===");
            ShapeManager manager = new ShapeManager();
            ShapeLogger logger = new ShapeLogger(output);
            manager.addObserver(logger);
            
            manager.addShape(circle);
            manager.addShape(rectangle);
            manager.updateShape(circle);
            
            // Test Decorator pattern
            System.out.println("\n=== Decorator Pattern ===");
            Shape decoratedCircle = new BorderDecorator(circle, "dashed", 1.0);
            System.out.println(decoratedCircle.getDescription());
            
            // Test Strategy pattern
            System.out.println("\n=== Strategy Pattern ===");
            AreaCalculator calculator = new AreaCalculator(new PreciseAreaCalculation());
            System.out.printf("Precise area: %.2f%n", calculator.calculateArea(rectangle));
            
            calculator.setStrategy(new ApproximateAreaCalculation());
            System.out.printf("Approximate area: %.2f%n", calculator.calculateArea(rectangle));
            
            // Test utility methods
            System.out.println("\n=== Utility Methods ===");
            double distance = ShapeUtils.calculateDistance(circle, rectangle);
            System.out.printf("Distance between shapes: %.2f%n", distance);
            
            List<Shape> sortedShapes = ShapeUtils.sortShapesByArea(Arrays.asList(circle, rectangle));
            System.out.println("Shapes sorted by area:");
            sortedShapes.forEach(shape -> 
                System.out.printf("  %s: %.2f%n", shape.getShapeType(), shape.calculateArea()));
            
            // Test reflection
            System.out.println("\n=== Reflection ===");
            ShapeInspector.inspectShape(circle);
            
            // Test exception handling
            System.out.println("\n=== Exception Handling ===");
            try {
                Circle invalidCircle = new Circle("black", -1.0, -5.0, new Point(0, 0));
                if (invalidCircle.getRadius() < 0) {
                    throw new InvalidShapeParameterException("radius", invalidCircle.getRadius());
                }
            } catch (InvalidShapeParameterException e) {
                System.out.println("Caught exception: " + e.getMessage());
            }
            
            System.out.println("\n=== Statistics ===");
            Map<String, DoubleSummaryStatistics> stats = 
                ShapeUtils.getAreaStatisticsByType(Arrays.asList(circle, rectangle));
            stats.forEach((type, stat) -> {
                System.out.printf("%s - Count: %d, Average: %.2f, Max: %.2f%n", 
                    type, stat.getCount(), stat.getAverage(), stat.getMax());
            });
            
        } catch (Exception e) {
            System.err.println("Error in demonstration: " + e.getMessage());
            e.printStackTrace();
        }
        
        System.out.println("\nAdvanced OOP demonstration completed!");
    }
}