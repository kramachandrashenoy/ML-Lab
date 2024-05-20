def list_operations():
    lst = []
    while True:
        print("\nList Operations:")
        print("1. Insert")
        print("2. Update")
        print("3. Delete")
        print("4. Display")
        print("5. Sort")
        print("6. Search")
        print("7. Exit")
        choice = int(input("Enter your choice: "))

        if choice == 1:
            element = input("Enter element to insert: ")
            lst.append(element)
        elif choice == 2:
            index = int(input("Enter index to update: "))
            element = input("Enter new element: ")
            if 0 <= index < len(lst):
                lst[index] = element
            else:
                print("Index out of range.")
        elif choice == 3:
            element = input("Enter element to delete: ")
            if element in lst:
                lst.remove(element)
            else:
                print("Element not found.")
        elif choice == 4:
            print("List:", lst)
        elif choice == 5:
            lst.sort()
            print("Sorted List:", lst)
        elif choice == 6:
            element = input("Enter element to search: ")
            if element in lst:
                print("Element found at index:", lst.index(element))
            else:
                print("Element not found.")
        elif choice == 7:
            break
        else:
            print("Invalid choice. Please try again.")

def tuple_operations():
    tpl = ()
    while True:
        print("\nTuple Operations:")
        print("1. Display")
        print("2. Exit")
        choice = int(input("Enter your choice: "))

        if choice == 1:
            print("Tuple:", tpl)
        elif choice == 2:
            break
        else:
            print("Invalid choice. Please try again.")

def set_operations():
    st = set()
    while True:
        print("\nSet Operations:")
        print("1. Insert")
        print("2. Delete")
        print("3. Display")
        print("4. Search")
        print("5. Exit")
        choice = int(input("Enter your choice: "))

        if choice == 1:
            element = input("Enter element to insert: ")
            st.add(element)
        elif choice == 2:
            element = input("Enter element to delete: ")
            st.discard(element)
        elif choice == 3:
            print("Set:", st)
        elif choice == 4:
            element = input("Enter element to search: ")
            if element in st:
                print("Element found.")
            else:
                print("Element not found.")
        elif choice == 5:
            break
        else:
            print("Invalid choice. Please try again.")

def dict_operations():
    dct = {}
    while True:
        print("\nDictionary Operations:")
        print("1. Insert/Update")
        print("2. Delete")
        print("3. Display")
        print("4. Search")
        print("5. Exit")
        choice = int(input("Enter your choice: "))

        if choice == 1:
            key = input("Enter key: ")
            value = input("Enter value: ")
            dct[key] = value
        elif choice == 2:
            key = input("Enter key to delete: ")
            if key in dct:
                del dct[key]
            else:
                print("Key not found.")
        elif choice == 3:
            print("Dictionary:", dct)
        elif choice == 4:
            key = input("Enter key to search: ")
            if key in dct:
                print("Key found. Value:", dct[key])
            else:
                print("Key not found.")
        elif choice == 5:
            break
        else:
            print("Invalid choice. Please try again.")

def main():
    while True:
        print("\nMain Menu:")
        print("1. List Operations")
        print("2. Tuple Operations")
        print("3. Set Operations")
        print("4. Dictionary Operations")
        print("5. Exit")
        choice = int(input("Enter your choice: "))

        if choice == 1:
            list_operations()
        elif choice == 2:
            tuple_operations()
        elif choice == 3:
            set_operations()
        elif choice == 4:
            dict_operations()
        elif choice == 5:
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
