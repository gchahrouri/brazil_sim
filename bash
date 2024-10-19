mkdir -p app/static/css app/templates

cat > app/__init__.py << EOL
from flask import Flask

app = Flask(__name__)

from app import routes
EOL
