import pytest 
import sys
import os
import requests 
import code.apicalls as calls



def test_should_pass():
    print("\nAlways True!")
    assert True


def test_get_googe_place_details():
    tests = [ 
        {'place_id': 'ChIJUTtvv9Tz2YkRhneTbRT-1mk', 'expected_name': 'Buried Acorn Brewery'},
        { 'place_id': 'ChIJl2h_-pjz2YkR-VUHD9dpOF0', 'expected_name': 'Meier’s Creek Brewing - Inner Harbor'},
    ]
    for t in tests:
        print(f"\nTESTING: test_get_googe_place_details({t['place_id']}) == {t['expected_name']}")
        place =  calls.get_google_place_details(t['place_id']) 
        assert place['result']['name'] == t['expected_name']


def test_get_googe_place_details_check_exception():
    # check for exception
    test = {'place_id': '12345', 'expected_name': 'Does not exist!'}
    try:
        place =  calls.get_google_place_details(test['place_id']) 
        assert False # We Expected an HTTPError
    except requests.exceptions.HTTPError as e:
        print(f"HTTPError: {e}")
        assert True


def test_get_azure_sentiment():
    tests = [
        {'text': 'I love programming!', 'expected_sentiment': 'positive'},
        {'text': 'I hate bugs.', 'expected_sentiment': 'negative'},
    ]
    for t in tests:
        print(f"\nTESTING: test_get_azure_sentiment({t['text']}) == {t['expected_sentiment']}")
        results = calls.get_azure_sentiment(t['text'])
        sentiment = results['results']['documents'][0]['sentiment']
        assert t['expected_sentiment'] == sentiment

def test_get_azure_key_phrase_extraction():
    tests = [
        {'text': 'Microsoft was founded by Bill Gates and Paul Allen.', 'expected_entities': ['Microsoft', 'Bill Gates', 'Paul Allen']},
        {'text': 'The Eiffel Tower is located in Paris.', 'expected_entities': ['The Eiffel Tower', 'Paris']},
    ]
    for t in tests:
        print(f"\nTESTING: get_azure_named_entity_recognition({t['text']}) == {t['expected_entities']}")
        results = calls.get_azure_key_phrase_extraction(t['text'])
        entities = results['results']['documents'][0]['keyPhrases']
        for e in entities:
            assert e in t['expected_entities']

def test_get_azure_named_entity_recognition():
    tests = [
        {'text': 'Microsoft was founded by Bill Gates and Paul Allen.', 'expected_entities': ['Microsoft', 'Bill Gates', 'Paul Allen']},
        {'text': 'The Eiffel Tower is located in Paris.', 'expected_entities': ['Eiffel Tower', 'Paris']},
    ]
    for t in tests:
        print(f"\nTESTING: get_azure_named_entity_recognition({t['text']}) == {t['expected_entities']}")
        results = calls.get_azure_named_entity_recognition(t['text'])
        entities = results['results']['documents'][0]['entities']
        for e in entities:
            assert e['text'] in t['expected_entities']


# IF YOU NEED TO DEBUG A TEST
# 1. Place a breakpoint on the line below
# 2. call the function you want to debug below the if statement
# Run this file with debugging
if __name__ == "__main__":
    test_should_pass()
