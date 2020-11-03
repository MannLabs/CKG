from passlib.hash import bcrypt
from graphdb_connector import connector

driver = connector.getGraphDatabaseConnectionConfiguration()


class User:
    def __init__(self, username):
        self.username = username

    def find(self):
        user = connector.find_node(driver, node_type="User", username=self.username)
        return user

    def register(self, password):
        if not self.find():
            return connector.create_node(driver, "User", username=self.username, password=bcrypt.encrypt(password))
        else:
            return False

    def verify_password(self, password):
        user = self.find()
        if user:
            return bcrypt.verify(password, user['password'])
        else:
            return False
