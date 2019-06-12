import webbrowser
import tweepy

def login():
    consumer_key = input('Consumer key: ').strip()
    consumer_secret = input('Consumer secret: ').strip()
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)

    webbrowser.open(auth.get_authorization_url())

    pin = input('Verification pin number from twitter.com: ').strip()

    token = auth.get_access_token(verifier=pin)
    api = tweepy.API(auth)

    return api, token

def pick_a_list(api, user):
    user_lists = api.lists_all(user.screen_name)
    for i, user_list in enumerate(user_lists):
        print(i, user_list.name)

    list_num = int(input('Enter the number of the group that you want to follow: '))

    return user_lists[list_num]

def follow_list(myuser, user_list):
    for member in api.list_members(owner_screen_name=myuser.screen_name, slug=user_list.slug):
        api.create_friendship(member.screen_name)

api, token = login()

user = api.get_user(input("Username of the account with the list: "))
selected_list = pick_a_list(api, user)

follow_list(user, selected_list)
