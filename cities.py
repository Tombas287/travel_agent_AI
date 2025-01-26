from libraries import st

def default_cities():
    cities = [
        "Mumbai", "Bangalore", "Delhi", "Chennai", "Kolkata", "Hyderabad",
        "New York", "Los Angeles", "London", "Paris", "Berlin", "Tokyo", "Sydney",
        "Shanghai", "Singapore", "Dubai", "Rio de Janeiro", "Cape Town", "Moscow",
        "Toronto", "Mexico City", "Cairo", "Buenos Aires", "Istanbul", "Seoul", "Bangkok"
    ]

    return cities

def check_email(user_id: str):
    email_list = ["maxbasumatry@gmail.com", "divesh@gmail.com"]
    if user_id in email_list:
        return True

    st.sidebar.error("Email ID need to entered")
    return False

