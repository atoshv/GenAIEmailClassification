from flask import Flask, request, jsonify
from email_classification import set_category_subcategory, load_mappings, VALID_CATEGORIES

app = Flask(__name__)

@app.route('/set_category', methods=['POST'])
def set_category():
    """API endpoint to add or update categories and sub-categories."""
    data = request.json
    category = data.get("category")
    subcategories = data.get("subcategories")

    if not category or not subcategories or not isinstance(subcategories, list):
        return jsonify({"error": "Invalid input. Provide a category and a list of sub-categories."}), 400

    if category not in VALID_CATEGORIES:
        return jsonify({
            "error": f"Invalid category. Must be one of: {', '.join(VALID_CATEGORIES)}",
            "valid_categories": list(VALID_CATEGORIES)
        }), 400

    try:
        updated_mappings = set_category_subcategory(category, subcategories)
        return jsonify({
            "message": f"Category '{category}' updated successfully.",
            "mappings": updated_mappings
        }), 200
    except Exception as e:
        app.logger.error(f"Error in set_category: {e}")
        return jsonify({"error": "An internal error has occurred."}), 500

@app.route('/get_mappings', methods=['GET'])
def get_mappings():
    """API endpoint to retrieve all mappings."""
    try:
        mappings = load_mappings()
        return jsonify({
            "mappings": mappings,
            "valid_categories": list(VALID_CATEGORIES)
        }), 200
    except Exception as e:
        app.logger.error(f"Error in get_mappings: {e}")
        return jsonify({"error": "An internal error has occurred."}), 500

import os

if __name__ == '__main__':
    debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() in ['true', '1', 't']
    app.run(debug=debug_mode)