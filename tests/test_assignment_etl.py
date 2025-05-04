import pytest 
import pandas as pd
import sys
import os 
import code.assignment_etl as etl


def test_should_pass():
    print("\nAlways True!")
    assert True


def test_reviews_step_output():
    file = etl.CACHE_REVIEWS_FILE
    lines = 10
    cols = ['place_id','name','author_name','rating','text']

    print(f"TESTING: {file} file exists")
    assert os.path.exists(file)

    print(f"TESTING: {file} read_csv, {lines} lines")
    df = pd.read_csv(file)
    assert len(df) ==  lines
    
    print(f"TESTING: {file} columns : {cols}")
    for c in df:
        assert c.lower() in cols
          
def test_sentiment_step_output():
    file = etl.CACHE_SENTIMENT_FILE

    lines = 80
    cols = [ c.strip().lower() for c in "place_id,name,author_name,rating,sentence_text,sentence_sentiment,confidenceScores.positive,confidenceScores.neutral,confidenceScores.negative".split(",")]

    print(f"TESTING: {file} file exists")
    assert os.path.exists(file)

    print(f"TESTING: {file} read_csv, {lines} lines")
    df = pd.read_csv(file)
    assert len(df) >=  lines
    
    print(f"TESTING: {file} columns : {cols}")
    for c in df:
        assert c.lower() in cols


def test_entity_exraction_step_file_in_cache():

    file =etl.CACHE_ENTITIES_FILE
    lines = 100
    cols = [ c.strip().lower() for c in "place_id,name,author_name,rating,sentence_text,sentence_sentiment,confidenceScores.positive,confidenceScores.neutral,confidenceScores.negative,entity_text,entity_category,entity_subcategory,confidenceScores.entity".split(",")]

    print(f"TESTING: {file} file exists")
    assert os.path.exists(file)

    print(f"TESTING: {file} read_csv, {lines} lines")
    df = pd.read_csv(file)
    assert len(df) >=  lines
    
    print(f"TESTING: {file} columns : {cols}")
    for c in df:
        assert c.lower() in cols


# IF YOU NEED TO DEBUG A TEST
# 1. Place a breakpoint on the line below
# 2. call the function you want to debug below the if statement
# Run this file with debugging
if __name__ == "__main__":
    test_should_pass()