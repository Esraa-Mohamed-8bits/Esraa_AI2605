import csv        
import os          
from datetime import datetime  

MANAGER_ID       = "manager"
MANAGER_PASSWORD = "admin123"


# User class 
class User:

    def __init__(self, user_id, name, password, phone, balance=0.0):
        self.user_id  = user_id         
        self.name     = name            
        self.password = password         
        self.phone    = phone            
        self.balance  = float(balance)   


# BankSystem class 
class BankSystem:

    def __init__(self, users_file="data.csv", transactions_file="transactions.csv"):
        self.users_file        = users_file
        self.transactions_file = transactions_file
        self.users = {}
        self._load_users()


    # LOGIN 

    def login_manager(self, user_id, password):
        return user_id == MANAGER_ID and password == MANAGER_PASSWORD

    def login_user(self, user_id, password):
        user = self.users.get(user_id)         
        if user is not None and user.password == password:
            return user
        return None


    # LOADING & SAVING DATA

    def _load_users(self):
        if not os.path.exists(self.users_file):
            return

        file = open(self.users_file, newline="", encoding="utf-8")
        reader = csv.DictReader(file)
        for row in reader:
            user = User(
                user_id  = row["user_id"],
                name     = row["name"],
                password = row["password"],
                phone    = row["phone"],
                balance  = row["balance"],
            )
            self.users[row["user_id"]] = user
        file.close()

    def _save_users(self):
        file = open(self.users_file, "w", newline="", encoding="utf-8")
        fieldnames = ["user_id", "name", "password", "phone", "balance"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for user in self.users.values():
            writer.writerow({
                "user_id":  user.user_id,
                "name":     user.name,
                "password": user.password,
                "phone":    user.phone,
                "balance":  user.balance,
            })
        file.close()

    def _save_transaction(self, user_id, action, amount, note=""):
        file_exists = os.path.exists(self.transactions_file)

        file = open(self.transactions_file, "a", newline="", encoding="utf-8")
        fieldnames = ["timestamp", "user_id", "action", "amount", "note"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "user_id":   user_id,
            "action":    action,
            "amount":    amount,
            "note":      note,
        })
        file.close()


    # ID GENERATION

    def _generate_id(self):
        if not self.users:
            return "U001"

        existing_numbers = []
        for uid in self.users.keys():
            try:
                existing_numbers.append(int(uid[1:]))
            except ValueError:
                pass

        if not existing_numbers:
            return "U001"

        next_number = max(existing_numbers) + 1
        return f"U{next_number:03d}"


    # ACCOUNT OPERATIONS

    def create_account(self, name, password, phone, balance=0.0):
        user_id = self._generate_id()
        new_user = User(user_id, name, password, phone, balance)
        self.users[user_id] = new_user
        self._save_users()
        self._save_transaction(user_id, "ACCOUNT_CREATED", balance, f"Account created for {name}")
        return user_id

    def delete_account(self, user_id):
        if user_id not in self.users:
            return False, "User not found."
        del self.users[user_id]
        self._save_users()
        self._save_transaction(user_id, "ACCOUNT_DELETED", 0, "Account deleted")
        return True, "Account deleted successfully."

    def get_user(self, user_id):
        return self.users.get(user_id)

    def get_all_users(self):
        return list(self.users.values())


    # UPDATE OPERATIONS

    def update_name(self, user_id, new_name):
        if user_id not in self.users:
            return False, "User not found."
        self.users[user_id].name = new_name
        self._save_users()
        return True, "Name updated successfully."

    def update_password(self, user_id, new_password):
        if user_id not in self.users:
            return False, "User not found."
        self.users[user_id].password = new_password
        self._save_users()
        return True, "Password updated successfully."

    def update_phone(self, user_id, new_phone):
        if user_id not in self.users:
            return False, "User not found."
        self.users[user_id].phone = new_phone
        self._save_users()
        return True, "Phone number updated successfully."

    def update_balance(self, user_id, new_balance):
        if user_id not in self.users:
            return False, "User not found."
        self.users[user_id].balance = float(new_balance)
        self._save_users()
        self._save_transaction(user_id, "BALANCE_UPDATED", new_balance,
                               "Balance edited directly by manager")
        return True, "Balance updated successfully."


    # BANKING OPERATIONS

    def deposit(self, user_id, amount):
        if user_id not in self.users:
            return False, "User not found."
        if amount <= 0:
            return False, "Amount must be greater than zero."
        self.users[user_id].balance += amount
        self._save_users()
        self._save_transaction(user_id, "DEPOSIT", amount)
        return True, f"Deposited ${amount:.2f} successfully."

    def withdraw(self, user_id, amount):
        if user_id not in self.users:
            return False, "User not found."
        if amount <= 0:
            return False, "Amount must be greater than zero."
        if self.users[user_id].balance < amount:
            return False, "Insufficient balance."
        self.users[user_id].balance -= amount
        self._save_users()
        self._save_transaction(user_id, "WITHDRAWAL", amount)
        return True, f"Withdrew ${amount:.2f} successfully."


    # TRANSACTION HISTORY

    def get_transactions(self, user_id=None):
        if not os.path.exists(self.transactions_file):
            return []

        all_transactions = []
        file = open(self.transactions_file, newline="", encoding="utf-8")
        reader = csv.DictReader(file)
        for row in reader:
            if user_id is None or row["user_id"] == user_id:
                all_transactions.append(row)
        file.close()

        return list(reversed(all_transactions))


    # CSV EXPORT

    def get_users_csv(self):
        lines = ["user_id,name,phone,balance"]
        for user in self.users.values():
            lines.append(f"{user.user_id},{user.name},{user.phone},{user.balance:.2f}")
        return "\n".join(lines)

    def get_transactions_csv(self):
        if not os.path.exists(self.transactions_file):
            return "timestamp,user_id,action,amount,note\n"
        file = open(self.transactions_file, "r", encoding="utf-8")
        content = file.read()
        file.close()
        return content