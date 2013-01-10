from nose.tools import *
import NAME

def setup():
    print "Hands up! This is a setup!"

def teardown():
    print "Did someone call for an exterminator?"

def test_basic():
    print "Rawr, this is a test."