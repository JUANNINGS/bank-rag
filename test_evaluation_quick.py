"""
Quick Test for Enhanced Evaluation System
Tests that all imports work and basic functionality is available
"""

print("🧪 Testing Enhanced Evaluation System...")
print("="*60)

# Test 1: Import all evaluation modules
print("\n[1/5] Testing imports...")
try:
    from evaluation import (
        ComprehensiveRAGEvaluator,
        RagasEvaluator,
        generate_comprehensive_test_queries,
        ComprehensiveMetrics,
        RetrievalMetrics,
        GenerationMetrics
    )
    print("  ✅ All evaluation modules imported successfully")
except Exception as e:
    print(f"  ❌ Import failed: {e}")
    exit(1)

# Test 2: Check test dataset
print("\n[2/5] Testing test dataset...")
try:
    test_queries = generate_comprehensive_test_queries()
    print(f"  ✅ Loaded {len(test_queries)} test queries")
    
    # Check structure
    first_query = test_queries[0]
    assert 'query' in first_query
    assert 'expected_sources' in first_query
    assert 'reference_answer' in first_query
    assert 'category' in first_query
    print(f"  ✅ Test query structure validated")
    
    # Count by category
    categories = {}
    for q in test_queries:
        cat = q['category']
        categories[cat] = categories.get(cat, 0) + 1
    
    print(f"\n  📊 Test coverage by category:")
    for cat, count in sorted(categories.items()):
        print(f"     {cat:15s}: {count} questions")
    
except Exception as e:
    print(f"  ❌ Test dataset failed: {e}")
    exit(1)

# Test 3: Check Ragas is available
print("\n[3/5] Testing Ragas availability...")
try:
    import ragas
    from ragas.metrics import faithfulness, answer_relevancy
    print(f"  ✅ Ragas installed (version: {ragas.__version__ if hasattr(ragas, '__version__') else 'unknown'})")
except Exception as e:
    print(f"  ⚠️  Ragas import warning: {e}")
    print(f"  ℹ️  You can still use traditional metrics")

# Test 4: Check config
print("\n[4/5] Testing configuration...")
try:
    from config import get_azure_config
    config = get_azure_config()
    
    if config.api_key and config.api_key != "":
        print("  ✅ Azure OpenAI API key configured")
    else:
        print("  ⚠️  Azure OpenAI API key not configured")
        print("  ℹ️  Make sure .env file is set up")
    
    if config.endpoint:
        print(f"  ✅ Endpoint: {config.endpoint[:30]}...")
    
    print(f"  ✅ GPT Deployment: {config.gpt_deployment}")
    print(f"  ✅ Embedding Deployment: {config.embedding_deployment}")
    
except Exception as e:
    print(f"  ❌ Config test failed: {e}")
    print(f"  ℹ️  Make sure .env file exists and is properly configured")

# Test 5: Sample test query
print("\n[5/5] Sample test query structure...")
try:
    sample = test_queries[0]
    print(f"\n  📝 Example Test Case:")
    print(f"     Category: {sample['category']}")
    print(f"     Query: {sample['query']}")
    print(f"     Expected Sources: {sample['expected_sources']}")
    print(f"     Reference Answer: {sample['reference_answer'][:80]}...")
    print(f"  ✅ Sample validated")
except Exception as e:
    print(f"  ❌ Sample test failed: {e}")

# Summary
print("\n" + "="*60)
print("✅ QUICK TEST COMPLETE!")
print("="*60)
print("\n📚 Next steps:")
print("   1. Ensure your .env file is configured with Azure OpenAI credentials")
print("   2. Run full evaluation: python3 evaluation.py")
print("   3. Check reports in ./tests/ directory")
print("\n💡 Tips:")
print("   - Full evaluation takes ~5-10 minutes for 30 questions")
print("   - Use use_ragas=False to skip LLM-based metrics (faster)")
print("   - Start with 5-10 test questions during development")
print("\n📖 See EVALUATION_GUIDE.md for detailed usage instructions")
print()







