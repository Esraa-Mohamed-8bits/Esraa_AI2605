import streamlit as st        
import pandas as pd             
from System import BankSystem


# PAGE CONFIGURATION  
st.set_page_config(
    page_title = "STREAMBank",
    layout     = "centered",
)


# Create one BankSystem object that exists for the whole session
bank = BankSystem()

# Session state variables keep track of who is logged in
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False   
    st.session_state.role      = None    
    st.session_state.user_id   = None    
    st.session_state.user_name = None    


def logout():
    st.session_state.logged_in = False
    st.session_state.role      = None
    st.session_state.user_id   = None
    st.session_state.user_name = None



# LOGIN PAGE

def show_login():
     
    st.title("Login to STREAMbank's system") 

    tab_login, tab_signin = st.tabs(["Login ", " Signin"])

    # login tab
    with tab_login:

        uid = st.text_input("User ID", placeholder="e.g.  U001   or   manager",key="login_uid")
        pwd = st.text_input("Password", type="password", key="login_pwd")

        if st.button("Login", use_container_width=True):

            # Check if it's the manager
            if bank.login_manager(uid, pwd):
                st.session_state.logged_in = True
                st.session_state.role      = "manager"
                st.session_state.user_name = "Manager"
                st.rerun()

            else:
                # Check if it's a regular user
                user = bank.login_user(uid, pwd)
                if user is not None:
                    st.session_state.logged_in = True
                    st.session_state.role      = "user"
                    st.session_state.user_id   = uid
                    st.session_state.user_name = user.name
                    st.rerun()
                else:
                    st.error("! Wrong User ID or password. Please try again.")

        

    # Signin tab
    with tab_signin:

        full_name = st.text_input("Full Name")

        col1, col2 = st.columns(2)
        with col1:
            new_pwd = st.text_input("Password", type="password")
        with col2:
            phone = st.text_input("Phone Number")

        init_balance = st.number_input("Initial Deposit ($)", min_value=0.0, step=10.0)

        if st.button("Create My Account", use_container_width=True):
            # Validate that nothing is empty
            if not full_name.strip():
                st.error("Please enter your full name.")
            elif not new_pwd.strip():
                st.error("Please enter a password.")
            elif not phone.strip():
                st.error("Please enter your phone number.")
            else:
                new_id = bank.create_account(
                    full_name.strip(), new_pwd.strip(),
                    phone.strip(), init_balance
                )
                st.success(f"Account created! Your User ID is **{new_id}**")
                st.info("!!!! Please save your User ID (you need it to login).")


# MANAGER DASHBOARD

def show_manager():

    # Sidebar
    with st.sidebar:
        st.text(f"Welcome back Manager to your STREAMbank!")       
        st.divider()

        # Menu
        page = st.selectbox(
            "Go to",
            options=[
                "Dashboard",
                "All Users",
                "Edit User",
                "Delete User",
                "All Transactions",
                "Download Data",
            ],
            label_visibility="collapsed",
        )

        st.divider()
        st.button("Logout", use_container_width=True, on_click= logout, key="manager_logout")



    # Dashboard
    if page == "Dashboard":
        st.title("Manager Dashboard")
        all_users = bank.get_all_users()
        all_txns  = bank.get_transactions()
        total_bal = sum(u.balance for u in all_users)

        # Three summary stat cards
        col1, col2,col3 = st.columns(3)   
        with col1:
            st.metric("Total Users", f"{len(all_users)}", delta="+active", delta_color="blue")
        with col2:
            st.metric("Total Balance", f"${total_bal:.2f}", delta="+deposits", delta_color="green")
        with col3:
            st.metric("Total Transactions", f"{len(all_txns)}", delta="+new", delta_color="red")

        st.markdown("#### Recent Transactions")
        if all_txns:
            # Show only the 10 most recent
            df = pd.DataFrame(all_txns[:10])
            st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No transactions recorded yet.")


    # All Users 
    elif page == "All Users":
        st.text("All registered accounts in the system. You can edit or delete any account from the sidebar menu.")
        all_users = bank.get_all_users()
        if not all_users:
            st.info("No accounts have been created yet.")
        else:
            rows = []
            for u in all_users:
                rows.append({
                    "User ID":    u.user_id,
                    "Name":       u.name,
                    "Phone":      u.phone,
                    "Balance ($)": f"{u.balance:.2f}",
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.caption(f"Total: {len(all_users)} account(s)")


    # Edit User 
    elif page == "Edit User":
        st.text("Enter the User ID of the account you want to edit. You can update their name, password, phone number, or balance.")
        uid = st.text_input("Enter the User ID you want to edit", key="edit_uid")

        if uid.strip():
            user = bank.get_user(uid.strip())

            if user is None:
                st.error("User not found. Please check the ID.")
            else:
                st.success(f"Found: **{user.name}** (ID: {user.user_id})")
                st.markdown("---")

                field = st.selectbox(
                    "Which field do you want to update?",
                    ["Name", "Password", "Phone", "Balance"],
                    key="edit_field",
                )

                if field == "Password":
                    new_value = st.text_input("New Password", type="password", key="edit_new_value")
                else:
                    new_value = st.text_input(f"New {field}", key=f"edit_new_value_{field.lower()}")

                if st.button("Save Changes"):
                    if not new_value.strip():
                        st.error("The new value cannot be empty.")
                    elif field == "Name":
                        ok, msg = bank.update_name(uid.strip(), new_value.strip())
                        if ok:
                            st.success(msg)
                        else:
                            st.error(msg)
                    elif field == "Password":
                        ok, msg = bank.update_password(uid.strip(), new_value.strip())
                        if ok:
                            st.success(msg)
                        else:
                            st.error(msg)
                    elif field == "Phone":
                        ok, msg = bank.update_phone(uid.strip(), new_value.strip())
                        if ok:
                            st.success(msg)
                        else:
                            st.error(msg)
                    elif field == "Balance":
                        try:
                            float_value = float(new_value.strip())   
                            if float_value < 0:                     
                                st.error("Balance cannot be negative.")
                            else:
                                ok, msg = bank.update_balance(uid.strip(), float_value)
                                if ok:
                                    st.success(msg)
                                else:
                                    st.error(msg)
                        except ValueError:
                            st.error("Please enter a valid number for the balance.")
                                            


    # Delete User 
    elif page == "Delete User":
        st.text("Enter the User ID of the account you want to delete. Be careful — this action cannot be undone and all data will be lost!")
        uid = st.text_input("Enter the User ID to delete", key="delete_uid")

        if uid.strip():
            user = bank.get_user(uid.strip())

            if user is None:
                st.error("User not found. Please check the ID.")
            else:
                st.warning(
                    f"!!!!! You are about to permanently delete the account of "
                    f"**{user.name}** (Balance: ${user.balance:.2f})"
                )

                confirmed = st.checkbox("I understand this cannot be undone.")

                if confirmed:
                    if st.button("Delete Account"):
                        ok, msg = bank.delete_account(uid.strip())
                        if ok:
                            st.success(msg)
                        else:
                            st.error(msg)


    # All Transactions
    elif page == "All Transactions":
        st.text("View all transactions made by all users. You can filter by User ID to see only one customer's history.")        
        filter_uid = st.text_input("Filter by User ID (leave empty to see all)", key="filter_uid")

        # Use None to get all, or the typed ID to filter
        search_id = filter_uid.strip() if filter_uid.strip() else None
        txns = bank.get_transactions(search_id)

        if not txns:
            st.info("No transactions found." if search_id is None else f"No transactions found for User ID '{search_id}'.")
        else:
            st.dataframe(pd.DataFrame(txns), use_container_width=True, hide_index=True)
            st.caption(f"{len(txns)} transaction(s) found.")


    # Download Data
    elif page == "Download Data":
        st.text("Download all users and transactions data as CSV files. You can open these files in Excel or any text editor.")
        st.write("Click a button below to download the data as a CSV file.")

        col1, col2 = st.columns(2)

        with col1:
            st.text("All accounts — passwords not included", )
            st.download_button(
                label      = "Download users.csv",
                data       = bank.get_users_csv(),
                file_name  = "users.csv",
                use_container_width = True, key="download_users")

        with col2:
            st.text("All transactions ever made")
            st.download_button(
                label      = "Download transactions.csv",
                data       = bank.get_transactions_csv(),
                file_name  = "transactions.csv",
                use_container_width = True, key="download_transactions")


# USER DASHBOARD

def show_user():

    uid  = st.session_state.user_id
    user = bank.get_user(uid)   

    # Sidebar 
    with st.sidebar:
        st.text(f"Welcome back {st.session_state.user_name} to your STREAMbank account!")
        st.divider()

        page = st.selectbox(
            "Go to",
            options=[
                "My Account",
                "Deposit",
                "Withdraw",
                "Update Info",
                "My Transactions",
            ],
            label_visibility="collapsed",
        )

        st.divider()
        st.button("Logout", use_container_width=True, on_click= logout, key="user_logout" )



    # My Account
    if page == "My Account":
        # Refresh user data in case something changed this session
        user = bank.get_user(uid)

        st.title("This is your account overview. You can see your current balance, how many transactions you've made, and your account details.")
        txn_count = len(bank.get_transactions(uid))

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Current Balance", f"${user.balance:.2f}")   
        with col2:
            st.metric("Transactions Made", f"{txn_count}")

        st.markdown("#### Account Details")
        st.markdown(f"- **User ID:** {user.user_id}")
        st.markdown(f"- **Name:** {user.name}")
        st.markdown(f"- **Phone:** {user.phone}")
        st.markdown(f"- **Password:** {'*' * len(user.password)} (hidden for security)")


    # Deposit 
    elif page == "Deposit":
        user = bank.get_user(uid)
        st.title("Deposit Funds")
        st.text("Enter the amount you want to deposit into your account. After confirming, your balance will be updated immediately.")
        st.metric("Current Balance", f"${user.balance:.2f}",delta_color="green")

        amount = st.number_input("Amount to deposit ($)", min_value=0.01, step=10.0, key="deposit_amount")

        if st.button("Confirm Deposit"):
            ok, msg = bank.deposit(uid, amount)
            if ok:
                st.success(msg)
                # Show the updated balance immediately
                updated_user = bank.get_user(uid)
                st.metric("New Balance", f"${updated_user.balance:.2f}",delta_color="green")
            else:
                st.error(msg)


    # Withdraw
    elif page == "Withdraw":
        user = bank.get_user(uid)
        st.text("Enter the amount you want to withdraw from your account. After confirming, your balance will be updated immediately.")
        st.metric("Current Balance", f"${user.balance:.2f}",delta_color="green")

        amount = st.number_input("Amount to withdraw ($)", min_value=0.01, step=10.0, key="withdraw_amount")

        if st.button("Confirm Withdrawal"):
            ok, msg = bank.withdraw(uid, amount)
            if ok:
                st.success(msg)
                updated_user = bank.get_user(uid)
                st.metric("New Balance", f"${updated_user.balance:.2f}",delta_color="red")
            else:
                st.error(msg)


    # Update Info 
    elif page == "Update Info":
        st.text("Update your personal information here. Changes will be saved immediately.")

        field = st.selectbox("What would you like to update?",["Name", "Password", "Phone"])

        if field == "Password":
            new_value = st.text_input("New Password", type="password", key="update_password")
        else:
            new_value = st.text_input(f"New {field}", key=f"update_{field.lower()}" )

        if st.button("Save Changes"):
            if not new_value.strip():
                st.error("The field cannot be empty.")
            else:
                if field == "Name":
                    ok, msg = bank.update_name(uid, new_value.strip())
                    if ok:
                        # Keep the sidebar name in same session state updated for consistency
                        st.session_state.user_name = new_value.strip()
                        st.success(msg)
                    else:
                            st.error(msg)
                elif field == "Password":
                    ok, msg = bank.update_password(uid, new_value.strip())
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)
                else:
                    ok, msg = bank.update_phone(uid, new_value.strip())
                    if ok:                        
                        st.success(msg)
                    else:                     
                        st.error(msg)


    # My Transactions 
    elif page == "My Transactions":
        st.text("View your transaction history here. You can see all deposits, withdrawals, and any notes you added to transactions.")
        
        txns = bank.get_transactions(uid)   

        if not txns:
            st.info("You have no transactions yet. Try making a deposit or withdrawal first!")
        else:
            rows = []
            for t in txns:
                rows.append({
                    "Date & Time": t["timestamp"],
                    "Action":      t["action"],
                    "Amount ($)":  f"{float(t['amount']):.2f}",
                    "Note":        t.get("note", ""),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.caption(f"Total: {len(txns)} transaction(s)")


# Deciding which page to show

if not st.session_state.logged_in:
    show_login()                              
elif st.session_state.role == "manager":
    show_manager()                             
else:
    show_user()                               