#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Basic data structures in C
typedef struct {
    int *data;
    size_t size;
    size_t capacity;
} dynamic_array_t;

// Dynamic array implementation
dynamic_array_t* create_array(size_t initial_capacity) {
    dynamic_array_t *array = malloc(sizeof(dynamic_array_t));
    if (!array) return NULL;
    
    array->data = malloc(sizeof(int) * initial_capacity);
    if (!array->data) {
        free(array);
        return NULL;
    }
    
    array->size = 0;
    array->capacity = initial_capacity;
    return array;
}

int array_push(dynamic_array_t *array, int value) {
    if (!array) return -1;
    
    if (array->size >= array->capacity) {
        size_t new_capacity = array->capacity * 2;
        int *new_data = realloc(array->data, sizeof(int) * new_capacity);
        if (!new_data) return -1;
        
        array->data = new_data;
        array->capacity = new_capacity;
    }
    
    array->data[array->size++] = value;
    return 0;
}

int array_get(dynamic_array_t *array, size_t index) {
    if (!array || index >= array->size) return -1;
    return array->data[index];
}

void array_destroy(dynamic_array_t *array) {
    if (array) {
        free(array->data);
        free(array);
    }
}

// Linked list implementation
typedef struct node {
    int data;
    struct node *next;
} node_t;

typedef struct {
    node_t *head;
    size_t size;
} linked_list_t;

linked_list_t* create_list() {
    linked_list_t *list = malloc(sizeof(linked_list_t));
    if (!list) return NULL;
    
    list->head = NULL;
    list->size = 0;
    return list;
}

int list_prepend(linked_list_t *list, int value) {
    if (!list) return -1;
    
    node_t *new_node = malloc(sizeof(node_t));
    if (!new_node) return -1;
    
    new_node->data = value;
    new_node->next = list->head;
    list->head = new_node;
    list->size++;
    return 0;
}

int list_find(linked_list_t *list, int value) {
    if (!list) return -1;
    
    node_t *current = list->head;
    int index = 0;
    
    while (current) {
        if (current->data == value) return index;
        current = current->next;
        index++;
    }
    
    return -1; // Not found
}

void list_destroy(linked_list_t *list) {
    if (!list) return;
    
    node_t *current = list->head;
    while (current) {
        node_t *next = current->next;
        free(current);
        current = next;
    }
    free(list);
}

// Hash table implementation (simple)
#define HASH_TABLE_SIZE 101

typedef struct hash_entry {
    char *key;
    int value;
    struct hash_entry *next;
} hash_entry_t;

typedef struct {
    hash_entry_t *buckets[HASH_TABLE_SIZE];
} hash_table_t;

unsigned int hash_function(const char *key) {
    unsigned int hash = 0;
    while (*key) {
        hash = (hash * 31) + *key;
        key++;
    }
    return hash % HASH_TABLE_SIZE;
}

hash_table_t* create_hash_table() {
    hash_table_t *table = malloc(sizeof(hash_table_t));
    if (!table) return NULL;
    
    for (int i = 0; i < HASH_TABLE_SIZE; i++) {
        table->buckets[i] = NULL;
    }
    return table;
}

int hash_table_set(hash_table_t *table, const char *key, int value) {
    if (!table || !key) return -1;
    
    unsigned int index = hash_function(key);
    hash_entry_t *entry = table->buckets[index];
    
    // Check if key exists
    while (entry) {
        if (strcmp(entry->key, key) == 0) {
            entry->value = value;
            return 0;
        }
        entry = entry->next;
    }
    
    // Create new entry
    hash_entry_t *new_entry = malloc(sizeof(hash_entry_t));
    if (!new_entry) return -1;
    
    new_entry->key = malloc(strlen(key) + 1);
    if (!new_entry->key) {
        free(new_entry);
        return -1;
    }
    
    strcpy(new_entry->key, key);
    new_entry->value = value;
    new_entry->next = table->buckets[index];
    table->buckets[index] = new_entry;
    
    return 0;
}

int hash_table_get(hash_table_t *table, const char *key, int *value) {
    if (!table || !key || !value) return -1;
    
    unsigned int index = hash_function(key);
    hash_entry_t *entry = table->buckets[index];
    
    while (entry) {
        if (strcmp(entry->key, key) == 0) {
            *value = entry->value;
            return 0;
        }
        entry = entry->next;
    }
    
    return -1; // Not found
}

void hash_table_destroy(hash_table_t *table) {
    if (!table) return;
    
    for (int i = 0; i < HASH_TABLE_SIZE; i++) {
        hash_entry_t *entry = table->buckets[i];
        while (entry) {
            hash_entry_t *next = entry->next;
            free(entry->key);
            free(entry);
            entry = next;
        }
    }
    free(table);
}

// Example usage and testing
int main() {
    printf("C Data Structures Examples\n");
    printf("==========================\n\n");
    
    // Test dynamic array
    printf("Testing Dynamic Array:\n");
    dynamic_array_t *array = create_array(2);
    array_push(array, 10);
    array_push(array, 20);
    array_push(array, 30); // This should trigger reallocation
    
    printf("Array elements: ");
    for (size_t i = 0; i < array->size; i++) {
        printf("%d ", array_get(array, i));
    }
    printf("\nArray size: %zu, capacity: %zu\n\n", array->size, array->capacity);
    array_destroy(array);
    
    // Test linked list
    printf("Testing Linked List:\n");
    linked_list_t *list = create_list();
    list_prepend(list, 100);
    list_prepend(list, 200);
    list_prepend(list, 300);
    
    printf("Finding 200 in list: index %d\n", list_find(list, 200));
    printf("List size: %zu\n\n", list->size);
    list_destroy(list);
    
    // Test hash table
    printf("Testing Hash Table:\n");
    hash_table_t *table = create_hash_table();
    hash_table_set(table, "apple", 5);
    hash_table_set(table, "banana", 3);
    hash_table_set(table, "orange", 8);
    
    int value;
    if (hash_table_get(table, "banana", &value) == 0) {
        printf("Value for 'banana': %d\n", value);
    }
    
    hash_table_destroy(table);
    
    printf("\nAll tests completed successfully!\n");
    return 0;
}