/**
 * @file cpp_references_pointers_guide.cpp
 * @brief Comprehensive guide to C++ references, pointers, lvalues, and rvalues
 *
 * This file explains in detail:
 *   1. Memory addresses and the concept of "location"
 *   2. Pointers (C-style and C++)
 *   3. References (lvalue references)
 *   4. Lvalues vs Rvalues
 *   5. Rvalue references (C++11)
 *   6. Move semantics
 *   7. Perfect forwarding
 *   8. Reference collapsing rules
 *   9. Common pitfalls and best practices
 */

#include <iostream>
#include <string>
#include <type_traits>
#include <utility> // for std::move, std::forward
#include <vector>

// ============================================================================
// PART 1: MEMORY ADDRESSES - THE FOUNDATION
// ============================================================================

void part1_memory_addresses() {
    std::cout << "\n=== PART 1: Memory Addresses ===\n\n";

    std::cout << "Every variable in C++ has:\n";
    std::cout << "  1. A VALUE (the data it holds)\n";
    std::cout << "  2. A MEMORY ADDRESS (where it lives in RAM)\n";
    std::cout << "  3. A TYPE (what kind of data it is)\n\n";

    int x = 42;
    std::cout << "Example: int x = 42;\n";
    std::cout << "  Value:         " << x << "\n";
    std::cout << "  Address:       " << &x << "\n"; // & = "address-of" operator
    std::cout << "  Type:          int\n";
    std::cout << "  Size in bytes: " << sizeof(x) << "\n\n";

    std::cout << "The & operator (address-of):\n";
    std::cout << "  &x means \"give me the address of x\"\n";
    std::cout << "  The address is a number (typically shown in hexadecimal)\n";
    std::cout << "  Think of it like a street address for your data\n\n";
}

// ============================================================================
// PART 2: POINTERS - "A VARIABLE THAT STORES AN ADDRESS"
// ============================================================================

void part2_pointers() {
    std::cout << "\n=== PART 2: Pointers ===\n\n";

    std::cout << "A POINTER is a variable that stores a memory address.\n\n";

    int x = 42;
    int* ptr = &x; // ptr is a pointer to int, storing the address of x

    std::cout << "Example:\n";
    std::cout << "  int x = 42;\n";
    std::cout << "  int* ptr = &x;  // ptr points to x\n\n";

    std::cout << "Visualization:\n";
    std::cout << "  Memory:\n";
    std::cout << "    [Address " << &x << "]  x = 42\n";
    std::cout << "    [Address " << &ptr << "]  ptr = " << ptr << " (points to x)\n\n";

    std::cout << "Key operations:\n";
    std::cout << "  1. &x        = address of x = " << &x << "\n";
    std::cout << "  2. ptr       = value of ptr (which is an address) = " << ptr << "\n";
    std::cout << "  3. *ptr      = dereference ptr (get value at address) = " << *ptr << "\n";
    std::cout << "  4. &ptr      = address of ptr itself = " << &ptr << "\n\n";

    std::cout << "The * has TWO meanings:\n";
    std::cout << "  1. In DECLARATION: int* ptr  means \"ptr is a pointer to int\"\n";
    std::cout << "  2. In EXPRESSION:  *ptr      means \"dereference\" (follow the pointer)\n\n";

    // Modifying through pointer
    std::cout << "Modifying through pointer:\n";
    *ptr = 100; // Change the value at the address ptr points to
    std::cout << "  *ptr = 100;\n";
    std::cout << "  Now x = " << x << "  (changed!)\n\n";

    // Null pointers
    std::cout << "Null pointers:\n";
    int* null_ptr = nullptr; // C++11 way
    std::cout << "  int* null_ptr = nullptr;  // Points to nothing\n";
    std::cout << "  null_ptr = " << null_ptr << "\n";
    std::cout << "  Dereferencing a null pointer (*null_ptr) would CRASH!\n\n";

    // Pointer arithmetic
    std::cout << "Pointer arithmetic (arrays):\n";
    int arr[5] = {10, 20, 30, 40, 50};
    int* p = arr; // Array name decays to pointer to first element
    std::cout << "  int arr[5] = {10, 20, 30, 40, 50};\n";
    std::cout << "  int* p = arr;\n";
    std::cout << "  p[0] = " << p[0] << "  (same as *p)\n";
    std::cout << "  p[1] = " << p[1] << "  (same as *(p+1))\n";
    std::cout << "  p+1 advances by sizeof(int) bytes, not 1 byte!\n\n";
}

// ============================================================================
// PART 3: REFERENCES - "AN ALIAS FOR ANOTHER VARIABLE"
// ============================================================================

void part3_references() {
    std::cout << "\n=== PART 3: References (Lvalue References) ===\n\n";

    std::cout << "A REFERENCE is an ALIAS - another name for an existing variable.\n\n";

    int x = 42;
    int& ref = x; // ref is a reference to x (another name for x)

    std::cout << "Example:\n";
    std::cout << "  int x = 42;\n";
    std::cout << "  int& ref = x;  // ref is another name for x\n\n";

    std::cout << "Key point: ref and x are THE SAME THING\n";
    std::cout << "  x = " << x << "\n";
    std::cout << "  ref = " << ref << "\n";
    std::cout << "  &x = " << &x << "\n";
    std::cout << "  &ref = " << &ref << "  (same address!)\n\n";

    ref = 100;
    std::cout << "After ref = 100:\n";
    std::cout << "  x = " << x << "  (changed because ref IS x)\n";
    std::cout << "  ref = " << ref << "\n\n";

    std::cout << "References vs Pointers:\n";
    std::cout << "  REFERENCE (int& ref)\n";
    std::cout << "    • Must be initialized when declared\n";
    std::cout << "    • Cannot be null\n";
    std::cout << "    • Cannot be reassigned (always refers to same object)\n";
    std::cout << "    • No special syntax needed to use (looks like normal variable)\n";
    std::cout << "    • Safer and cleaner\n\n";

    std::cout << "  POINTER (int* ptr)\n";
    std::cout << "    • Can be uninitialized\n";
    std::cout << "    • Can be null\n";
    std::cout << "    • Can be reassigned to point to different objects\n";
    std::cout << "    • Need * to dereference\n";
    std::cout << "    • More flexible but more dangerous\n\n";

    // Common use: function parameters
    std::cout << "Common use - pass by reference:\n";
    std::cout << "  void increment(int& n) { n++; }  // Modifies original\n";
    std::cout << "  void print(const int& n) { ... } // Read-only, no copy\n\n";
}

// ============================================================================
// PART 4: LVALUES and RVALUES
// ============================================================================

void part4_lvalues_rvalues() {
    std::cout << "\n=== PART 4: Lvalues and Rvalues ===\n\n";

    std::cout << "This is where it gets interesting!\n\n";

    std::cout << "LVALUE (locator value):\n";
    std::cout << "  • Has a persistent memory address (you can take its address with &)\n";
    std::cout << "  • Can appear on LEFT side of assignment\n";
    std::cout << "  • Examples: variables, array elements, dereferenced pointers\n\n";

    std::cout << "RVALUE (read value):\n";
    std::cout << "  • Temporary value without persistent address\n";
    std::cout << "  • Can only appear on RIGHT side of assignment\n";
    std::cout << "  • Examples: literals, temporary objects, results of expressions\n\n";

    int x = 42;
    int y = 10;

    std::cout << "Examples:\n";
    std::cout << "  x = 42;        // x is lvalue, 42 is rvalue\n";
    std::cout << "  x + y = 10;    // ERROR! x+y is rvalue (temporary), can't assign to it\n";
    std::cout << "  &x;            // OK! x is lvalue\n";
    std::cout << "  &42;           // ERROR! 42 is rvalue, no address\n";
    std::cout << "  &(x + y);      // ERROR! x+y is rvalue\n\n";

    // Detailed examples
    std::cout << "More examples:\n";
    std::cout << "  LVALUES:\n";
    std::cout << "    • x, y           (variables)\n";
    std::cout << "    • arr[0]         (array element)\n";
    std::cout << "    • *ptr           (dereferenced pointer)\n";
    std::cout << "    • ++x            (pre-increment returns lvalue!)\n";
    std::cout << "    • str.data()     (function returning reference)\n\n";

    std::cout << "  RVALUES:\n";
    std::cout << "    • 42, 3.14, 'a'  (literals)\n";
    std::cout << "    • x + y          (temporary result)\n";
    std::cout << "    • x++            (post-increment returns rvalue!)\n";
    std::cout << "    • foo()          (function returning by value)\n";
    std::cout << "    • std::string(\"hello\")  (temporary object)\n\n";

    std::cout << "Key difference - pre vs post increment:\n";
    int a = 5;
    int& ref1 = ++a; // OK! ++a returns lvalue (a itself)
    // int& ref2 = a++;  // ERROR! a++ returns rvalue (old value as temporary)
    std::cout << "  int& ref1 = ++a;  // OK! ++a is lvalue\n";
    std::cout << "  int& ref2 = a++;  // ERROR! a++ is rvalue\n\n";
}

// ============================================================================
// PART 5: RVALUE REFERENCES (C++11)
// ============================================================================

void part5_rvalue_references() {
    std::cout << "\n=== PART 5: Rvalue References (C++11) ===\n\n";

    std::cout << "C++11 introduced RVALUE REFERENCES: T&&\n\n";

    std::cout << "Syntax:\n";
    std::cout << "  T&   = lvalue reference (binds to lvalues)\n";
    std::cout << "  T&&  = rvalue reference (binds to rvalues)\n\n";

    int x = 42;
    int& lref = x; // lvalue reference to x
    // int& lref2 = 42;      // ERROR! Can't bind lvalue ref to rvalue
    int&& rref = 42; // rvalue reference to temporary
    // int&& rref2 = x;      // ERROR! Can't bind rvalue ref to lvalue

    std::cout << "Examples:\n";
    std::cout << "  int x = 42;\n";
    std::cout << "  int& lref = x;      // OK\n";
    std::cout << "  int& lref2 = 42;    // ERROR! rvalue 42 can't bind to lvalue ref\n";
    std::cout << "  int&& rref = 42;    // OK! rvalue ref binds to rvalue\n";
    std::cout << "  int&& rref2 = x;    // ERROR! lvalue x can't bind to rvalue ref\n\n";

    std::cout << "const lvalue reference (special case):\n";
    const int& const_lref = 42; // OK! const lvalue ref can bind to rvalue
    std::cout << "  const int& const_lref = 42;  // OK! Extends lifetime of temporary\n\n";

    std::cout << "WHY rvalue references?\n";
    std::cout << "  Answer: MOVE SEMANTICS!\n";
    std::cout << "  Instead of copying temporary objects, we can STEAL their resources.\n\n";
}

// ============================================================================
// PART 6: MOVE SEMANTICS
// ============================================================================

class MyString {
private:
    char* data;
    size_t len;

public:
    // Constructor
    MyString(const char* str = "") {
        len = std::strlen(str);
        data = new char[len + 1];
        std::strcpy(data, str);
        std::cout << "  [Constructor] Created \"" << data << "\" at " << (void*)data << "\n";
    }

    // Copy constructor (expensive!)
    MyString(const MyString& other) {
        len = other.len;
        data = new char[len + 1];
        std::strcpy(data, other.data);
        std::cout << "  [Copy Constructor] Copied \"" << data << "\" to new memory " << (void*)data
                  << "\n";
    }

    // Move constructor (cheap!)
    MyString(MyString&& other) noexcept {
        data = other.data; // Steal the pointer
        len = other.len;
        other.data = nullptr; // Leave other in valid state
        other.len = 0;
        std::cout << "  [Move Constructor] Stole \"" << data << "\" from " << (void*)&other << "\n";
    }

    // Copy assignment
    MyString& operator=(const MyString& other) {
        if (this != &other) {
            delete[] data;
            len = other.len;
            data = new char[len + 1];
            std::strcpy(data, other.data);
            std::cout << "  [Copy Assignment] Copied \"" << data << "\"\n";
        }
        return *this;
    }

    // Move assignment
    MyString& operator=(MyString&& other) noexcept {
        if (this != &other) {
            delete[] data;
            data = other.data;
            len = other.len;
            other.data = nullptr;
            other.len = 0;
            std::cout << "  [Move Assignment] Stole \"" << data << "\"\n";
        }
        return *this;
    }

    ~MyString() {
        if (data) {
            std::cout << "  [Destructor] Deleting \"" << data << "\" at " << (void*)data << "\n";
        } else {
            std::cout << "  [Destructor] Already moved (null)\n";
        }
        delete[] data;
    }

    const char* c_str() const { return data ? data : ""; }
};

void part6_move_semantics() {
    std::cout << "\n=== PART 6: Move Semantics ===\n\n";

    std::cout << "Problem: Copying large objects is expensive.\n";
    std::cout << "Solution: MOVE instead of COPY when dealing with temporaries.\n\n";

    std::cout << "Example 1: Copy vs Move\n";
    std::cout << "----------------------------------------\n";
    MyString s1("Hello");
    std::cout << "\nMyString s2 = s1;  // Copy (s1 is lvalue)\n";
    MyString s2 = s1;

    std::cout << "\nMyString s3 = MyString(\"World\");  // Move (temporary is rvalue)\n";
    MyString s3 = MyString("World");

    std::cout << "\n----------------------------------------\n\n";

    std::cout << "std::move() - casting lvalue to rvalue:\n";
    std::cout << "  std::move(x) is just static_cast<T&&>(x)\n";
    std::cout << "  It doesn't actually move anything - just casts!\n\n";

    MyString s4("Foo");
    std::cout << "MyString s5 = std::move(s4);  // Force move\n";
    MyString s5 = std::move(s4);
    std::cout << "After move, s4 is in 'valid but unspecified state'\n";
    std::cout << "s4.c_str() = \"" << s4.c_str() << "\" (empty/null)\n\n";

    std::cout << "Move semantics in standard library:\n";
    std::cout << "  std::vector, std::string, std::unique_ptr all support move\n";
    std::cout << "  Passing by value + move can be efficient for large objects\n\n";
}

// ============================================================================
// PART 7: PERFECT FORWARDING
// ============================================================================

// Example: A wrapper function that forwards arguments
template <typename T>
void process(T&& arg) {
    // std::forward preserves lvalue/rvalue nature
    std::cout << "  Received: " << (std::is_lvalue_reference<T>::value ? "lvalue" : "rvalue")
              << "\n";
}

void part7_perfect_forwarding() {
    std::cout << "\n=== PART 7: Perfect Forwarding ===\n\n";

    std::cout << "Problem: How to forward arguments preserving their lvalue/rvalue nature?\n\n";

    std::cout << "Universal/Forwarding reference: T&&\n";
    std::cout << "  When T is a template parameter, T&& is SPECIAL:\n";
    std::cout << "    • If passed lvalue, T deduces to T&   (reference collapsing)\n";
    std::cout << "    • If passed rvalue, T deduces to T\n\n";

    std::cout << "Reference collapsing rules:\n";
    std::cout << "  T& &   → T&   (lvalue ref to lvalue ref = lvalue ref)\n";
    std::cout << "  T& &&  → T&   (rvalue ref to lvalue ref = lvalue ref)\n";
    std::cout << "  T&& &  → T&   (lvalue ref to rvalue ref = lvalue ref)\n";
    std::cout << "  T&& && → T&&  (rvalue ref to rvalue ref = rvalue ref)\n\n";

    int x = 42;
    std::cout << "Example:\n";
    std::cout << "process(x):         ";
    process(x); // T deduced as int&, T&& becomes int& && = int&
    std::cout << "process(42):        ";
    process(42); // T deduced as int, T&& is int&&

    std::cout << "\nstd::forward<T>(arg):\n";
    std::cout << "  • If T is lvalue reference, returns lvalue reference\n";
    std::cout << "  • If T is not reference, returns rvalue reference\n";
    std::cout << "  • Used in template functions to forward arguments perfectly\n\n";

    std::cout << "Typical use:\n";
    std::cout << "  template<typename... Args>\n";
    std::cout << "  void wrapper(Args&&... args) {\n";
    std::cout << "      target(std::forward<Args>(args)...);\n";
    std::cout << "  }\n\n";
}

// ============================================================================
// PART 8: COMMON PITFALLS
// ============================================================================

void part8_pitfalls() {
    std::cout << "\n=== PART 8: Common Pitfalls ===\n\n";

    std::cout << "1. Dangling references:\n";
    std::cout << "   int& getRef() {\n";
    std::cout << "       int x = 42;\n";
    std::cout << "       return x;  // DANGER! x dies when function returns\n";
    std::cout << "   }\n";
    std::cout << "   Solution: Return by value or use static/global variable\n\n";

    std::cout << "2. Dangling pointers:\n";
    std::cout << "   int* ptr = new int(42);\n";
    std::cout << "   delete ptr;\n";
    std::cout << "   *ptr = 10;  // DANGER! ptr points to freed memory\n";
    std::cout << "   Solution: Set ptr = nullptr after delete, use smart pointers\n\n";

    std::cout << "3. Moving from lvalue without std::move:\n";
    std::cout << "   std::vector<int> v1 = {1,2,3};\n";
    std::cout << "   std::vector<int> v2 = v1;           // Copy\n";
    std::cout << "   std::vector<int> v3 = std::move(v1); // Move\n\n";

    std::cout << "4. Using moved-from object:\n";
    std::cout << "   std::string s1 = \"hello\";\n";
    std::cout << "   std::string s2 = std::move(s1);\n";
    std::cout << "   std::cout << s1;  // DANGER! s1 is in valid but unspecified state\n";
    std::cout << "   Solution: Don't use s1 after moving (or reassign it first)\n\n";

    std::cout << "5. Returning rvalue reference:\n";
    std::cout << "   int&& bad() {\n";
    std::cout << "       int x = 42;\n";
    std::cout << "       return std::move(x);  // WRONG! Still returns reference to local\n";
    std::cout << "   }\n";
    std::cout << "   Solution: Just return by value, compiler will use move anyway\n\n";

    std::cout << "6. const rvalue reference (almost never useful):\n";
    std::cout << "   void func(const T&& arg);  // Rarely makes sense\n";
    std::cout << "   Can't move from const rvalue (move requires modifying source)\n\n";
}

// ============================================================================
// PART 9: PRACTICAL GUIDELINES
// ============================================================================

void part9_guidelines() {
    std::cout << "\n=== PART 9: Practical Guidelines ===\n\n";

    std::cout << "WHEN TO USE WHAT:\n\n";

    std::cout << "Pass by value (T):\n";
    std::cout << "  • Small types (int, double, pointers)\n";
    std::cout << "  • When you need a copy anyway\n";
    std::cout << "  Example: void set(std::string s) { data = std::move(s); }\n\n";

    std::cout << "Pass by const lvalue reference (const T&):\n";
    std::cout << "  • Large objects you don't want to copy\n";
    std::cout << "  • Read-only access\n";
    std::cout << "  Example: void print(const std::vector<int>& v) { ... }\n\n";

    std::cout << "Pass by lvalue reference (T&):\n";
    std::cout << "  • When function modifies the argument\n";
    std::cout << "  • Output parameters\n";
    std::cout << "  Example: void increment(int& n) { n++; }\n\n";

    std::cout << "Pass by rvalue reference (T&&):\n";
    std::cout << "  • Move constructors/assignment operators\n";
    std::cout << "  • Taking ownership of temporaries\n";
    std::cout << "  Example: MyClass(MyClass&& other) noexcept { ... }\n\n";

    std::cout << "Pass by forwarding reference (T&& with template):\n";
    std::cout << "  • Perfect forwarding in templates\n";
    std::cout << "  • Factory functions, wrappers\n";
    std::cout << "  Example: template<typename T> void wrapper(T&& arg) { ... }\n\n";

    std::cout << "RETURN TYPES:\n\n";
    std::cout << "  Return by value:       For most cases, let RVO/move work\n";
    std::cout << "  Return const ref:      For class members (getters)\n";
    std::cout << "  Return non-const ref:  For operators like [] or assignment\n";
    std::cout << "  DON'T return rvalue ref: Compiler knows when to move\n\n";

    std::cout << "MODERN C++ RULES OF THUMB:\n\n";
    std::cout << "  1. Prefer references over pointers when possible\n";
    std::cout << "  2. Use const& for read-only large objects\n";
    std::cout << "  3. Implement move constructor/assignment for classes with resources\n";
    std::cout << "  4. Use std::move when transferring ownership\n";
    std::cout << "  5. Use std::forward in forwarding templates\n";
    std::cout << "  6. Prefer smart pointers (unique_ptr, shared_ptr) over raw pointers\n";
    std::cout << "  7. Mark move operations noexcept when possible\n";
    std::cout << "  8. Don't return std::move(local_var) - just return local_var\n\n";
}

// ============================================================================
// PART 10: VISUAL SUMMARY
// ============================================================================

void part10_visual_summary() {
    std::cout << "\n=== PART 10: Visual Summary ===\n\n";

    std::cout << "┌─────────────────────────────────────────────────────────────┐\n";
    std::cout << "│                    VALUE CATEGORIES                         │\n";
    std::cout << "├─────────────────────────────────────────────────────────────┤\n";
    std::cout << "│                                                             │\n";
    std::cout << "│     LVALUE                         RVALUE                   │\n";
    std::cout << "│  (has address)                  (temporary)                 │\n";
    std::cout << "│                                                             │\n";
    std::cout << "│   • Variables                   • Literals (42, \"hi\")      │\n";
    std::cout << "│   • Function names              • Temporaries               │\n";
    std::cout << "│   • Array elements              • x + y                     │\n";
    std::cout << "│   • *ptr                        • x++                       │\n";
    std::cout << "│   • ++x                         • foo() [by value]          │\n";
    std::cout << "│                                                             │\n";
    std::cout << "└─────────────────────────────────────────────────────────────┘\n\n";

    std::cout << "┌─────────────────────────────────────────────────────────────┐\n";
    std::cout << "│                    REFERENCE TYPES                          │\n";
    std::cout << "├─────────────────────────────────────────────────────────────┤\n";
    std::cout << "│                                                             │\n";
    std::cout << "│  int&              Lvalue reference    (binds to lvalue)    │\n";
    std::cout << "│  const int&        Const lvalue ref    (binds to anything)  │\n";
    std::cout << "│  int&&             Rvalue reference    (binds to rvalue)    │\n";
    std::cout << "│  T&& (template)    Forwarding ref      (binds to anything)  │\n";
    std::cout << "│                                                             │\n";
    std::cout << "└─────────────────────────────────────────────────────────────┘\n\n";

    std::cout << "┌─────────────────────────────────────────────────────────────┐\n";
    std::cout << "│                 WHAT BINDS TO WHAT?                         │\n";
    std::cout << "├─────────────────────────────────────────────────────────────┤\n";
    std::cout << "│                    Lvalue x      Rvalue 42                  │\n";
    std::cout << "│                    ─────────────────────────                │\n";
    std::cout << "│  int&              ✓             ✗                          │\n";
    std::cout << "│  const int&        ✓             ✓                          │\n";
    std::cout << "│  int&&             ✗             ✓                          │\n";
    std::cout << "│  const int&&       ✗             ✓ (rare)                   │\n";
    std::cout << "│                                                             │\n";
    std::cout << "└─────────────────────────────────────────────────────────────┘\n\n";
}

// ============================================================================
// MAIN
// ============================================================================

int main() {
    std::cout << "╔═════════════════════════════════════════════════════════════╗\n";
    std::cout << "║   C++ POINTERS, REFERENCES, LVALUES & RVALUES GUIDE        ║\n";
    std::cout << "╚═════════════════════════════════════════════════════════════╝\n";

    part1_memory_addresses();
    part2_pointers();
    part3_references();
    part4_lvalues_rvalues();
    part5_rvalue_references();
    part6_move_semantics();
    part7_perfect_forwarding();
    part8_pitfalls();
    part9_guidelines();
    part10_visual_summary();

    std::cout << "\n╔═════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                      END OF GUIDE                           ║\n";
    std::cout << "╚═════════════════════════════════════════════════════════════╝\n";

    return 0;
}
