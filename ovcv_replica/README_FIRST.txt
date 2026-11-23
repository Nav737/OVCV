START HERE

1. Read README_32076VAE3.md for the bond terms, source references, assumptions and limitations.
2. Open bond_32076vae3_config.json and replace all fields marked as illustrative with Bloomberg values from one consistent valuation timestamp.
3. Run: python -m pip install -r requirements.txt
4. Run: python example_32076vae3.py
5. Review outputs_32076vae3/summary_32076vae3.csv and sensitivity_32076vae3.csv.
6. Run tests: python -m unittest -v

Important: The generated research price is not a Bloomberg price. Market curve / volatility / credit inputs are placeholders and several path-dependent contract features are not yet included in the core PDE. See README_32076VAE3.md before using any output.
