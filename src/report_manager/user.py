from passlib.hash import bcrypt
from graphdb_connector import connector
from graphdb_builder.builder import create_user

driver = connector.getGraphDatabaseConnectionConfiguration()


class User:
    def __init__(self, username, name=None, surname=None, acronym=None, affiliation=None, email=None, alternative_email=None, phone=None, image=None, expiration_date=365, role='reader'):
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
        self.expiration_date = expiration_date
        self.role = role

    def to_dict(self):
        return {'username': self.username,
                'password': self.password,
                'name': self.name,
                'surname': self.surname,
                'acronym': self.acronym,
                'affiliation': self.affiliation,
                'email': self.email,
                'secondinary_email': self.email,
                'phone_number': self.phone_number,
                'image': self.image,
                'expiration_date': self.expiration_date,
                'role': self.role
                }

    def generate_initial_password(self):
        return bcrypt.encrypt(self.username)

    def find(self):
        user = connector.find_node(driver, node_type="User", username=self.username)
        return user

    def register(self):
        result = False
        if not self.find():
            result = create_user.create_user_from_dict(driver, self.to_dict())
            if result is not None:
                result = True

        return result

    def verify_password(self, password):
        user = self.find()
        if user:
            return bcrypt.verify(password, user['password'])
        else:
            return False
