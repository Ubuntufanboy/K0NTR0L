import sys
import urllib.request
import urllib.error

def display_tos():
    tos_text = """
    ================================================================
                        TERMS OF SERVICE
    ================================================================
    1. The Server Owner is NOT responsible for any content uploaded
       by the client to this server.
    2. Any illegal content uploaded to this server will result in
       IMMEDIATE termination of access and potential reporting to
       relevant authorities.
    3. Seriously, don't be weird. Thanks for helping me gather data!
    ================================================================
    """
    print(tos_text)

def main():
    display_tos()

    while True:
        response = input("Do you accept these terms? (yes/no): ").lower().strip()

        if response in ['y', 'yes']:
            print("\nTOS Accepted. Connecting to server...")
            break
        elif response in ['n', 'no']:
            print("\nTOS Declined. Exiting program.")
            sys.exit(0)
        else:
            print("Invalid input. Please type 'yes' or 'no'.")

    try:
        with urllib.request.urlopen(SERVER_URL) as response:
            server_msg = response.read().decode('utf-8')
            print(f"Status Code: {response.getcode()}")
            print("Thank you! You will be getting an API key from apolloiscool shortly! Sit tight!")

    except urllib.error.URLError as e:
        print(f"Error: Could not connect to {SERVER_URL}")
        print(f"Reason: {e.reason}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    SERVER_URL = input("DM apolloiscool on discord to get your server to autheticate and recieve your API key\n\nEnter Server URL here: ")
