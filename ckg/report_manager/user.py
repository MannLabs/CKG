from datetime import datetime, timedelta
from passlib.hash import bcrypt
from ckg.graphdb_connector import connector
from ckg.graphdb_builder.builder import create_user


class User:
    def __init__(self, username, name=None, surname=None, acronym=None, affiliation=None, email=None, alternative_email=None, phone=None, image=None, expiration_date=365, role='reader'):
        self.id = None
        self.username = username
        self.name = name
        self.surname = surname
        self.acronym = acronym
        self.affiliation = affiliation
        self.email = email
        self.secondary_email = alternative_email
        self.phone_number = phone
        self.image = image
        self.password = self.generate_initial_password()
        self.expiration_date = datetime.today() + timedelta(days=expiration_date)
        self.rolename = role

    def to_dict(self):
        return {'ID': self.id,
                'username': self.username,
                'password': self.password,
                'name': self.name,
                'surname': self.surname,
                'acronym': self.acronym,
                'affiliation': self.affiliation,
                'email': self.email,
                'secondary_email': self.email,
                'phone_number': self.phone_number,
                'image': self.image,
                'expiration_date': self.expiration_date,
                'rolename': self.rolename
                }

    def generate_initial_password(self):
        return bcrypt.encrypt(self.username)

    def find(self):
        user = None
        driver = connector.getGraphDatabaseConnectionConfiguration()
        if driver is not None:
            user = connector.find_node(driver, node_type="User", parameters={"username": self.username})
        return user

    def validate_user(self):
        user = None
        email = None
        driver = connector.getGraphDatabaseConnectionConfiguration()
        if driver is not None:
            user = connector.find_node(driver, node_type="User", parameters={"username": self.username})
            email = connector.find_node(driver, node_type="User", parameters={'email': self.email})

        return user is None and email is None

    def register(self):
        result = False
        driver = connector.getGraphDatabaseConnectionConfiguration()
        if driver is None:
            result = 'error_msg'
        else:
            found = self.find()
            if found is not None:
                result = "error_exists"
            elif self.validate_user():
                result = create_user.create_user_from_dict(driver, self.to_dict())
                if result is not None:
                    result = 'ok'
                else:
                    result = 'error_database'
            else:
                result = 'error_email'

        return result

    def verify_password(self, password):
        user = self.find()
        if user:
            return bcrypt.verify(password, user['password'])
        else:
            return False
