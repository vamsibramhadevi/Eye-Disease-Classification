import streamlit as st

def main():
    st.title('Simple Streamlit App')
    
    # Add a text input widget
    user_input = st.text_input("Enter some text:", "Type here...")
    
    # Echo the user's input
    st.write("You entered:", user_input)

if __name__ == "__main__":
    main()
