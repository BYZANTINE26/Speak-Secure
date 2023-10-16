import glob
import os

def delete_user():
    name = input("Enter the name of the user you want to delete: ") # TODO: check if user exists
    [os.remove(path) for path in glob.glob('../voice_database/' + name + '/*')]
    os.rmdir('../voice_database/' + name)
    os.remove('../gmm_models/' + name + '.gmm')