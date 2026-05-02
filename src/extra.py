import shap

# Initialize the SHAP explainer
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(test_feat[features])

# Plot 1: Summary Plot (Which features move the needle most?)
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, test_feat[features], show=False)
plt.title("SHAP Value Analysis: Sentiment Impact on Volatility")
plt.savefig('shap_summary.png')
plt.show()