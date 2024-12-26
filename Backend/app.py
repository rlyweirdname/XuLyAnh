from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import base64
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import numpy as np
import traceback

app = Flask(__name__)
CORS(app)

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        # Using specific layers from both implementations
        self.req_features = ['0', '5', '10', '19', '28']  # Style layers
        self.content_layer = '21'  # Content layer (conv4_2)
        self.model = models.vgg19(pretrained=True).features[:29]
        
    def forward(self, x):
        features = {}
        for layer_num, layer in enumerate(self.model):
            x = layer(x)
            if str(layer_num) in self.req_features or str(layer_num) == self.content_layer:
                features[str(layer_num)] = x
        return features

def load_image(image_bytes, imsize=512):
    try:
        image = Image.open(io.BytesIO(image_bytes))
        if image.mode not in ('RGB', 'RGBA'):
            image = image.convert('RGB')
        
        loader = transforms.Compose([
            transforms.Resize(imsize),
            transforms.ToTensor(),
        ])
        
        tensor = loader(image).unsqueeze(0)
        return tensor.to(device)
        
    except Exception as e:
        raise ValueError(f"Error processing image: {str(e)}")

def tensor_to_base64(tensor):
    try:
        # Convert to image
        tensor = tensor.squeeze(0).cpu().detach().numpy()
        tensor = np.clip(tensor, 0, 1)
        tensor = (tensor * 255).astype(np.uint8)
        image = Image.fromarray(tensor.transpose(1, 2, 0))
        
        # Convert to base64
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=90)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
        
    except Exception as e:
        raise ValueError(f"Error converting tensor to base64: {str(e)}")

def calc_content_loss(gen_feat, orig_feat):
    return torch.mean((gen_feat - orig_feat) ** 2)

def calc_style_loss(gen, style):
    batch_size, channel, height, width = gen.shape
    G = torch.mm(gen.view(channel, height * width), gen.view(channel, height * width).t())
    A = torch.mm(style.view(channel, height * width), style.view(channel, height * width).t())
    return torch.mean((G - A) ** 2)

def calculate_loss(gen_features, orig_features, style_features, alpha=8, beta=70):
    style_loss = content_loss = 0
    
    content_loss = calc_content_loss(gen_features['21'], orig_features['21'])
    
    style_layers = ['0', '5', '10', '19', '28']
    for layer in style_layers:
        style_loss += calc_style_loss(gen_features[layer], style_features[layer])
    
    total_loss = alpha * content_loss + beta * style_loss
    return total_loss, content_loss, style_loss

def style_transfer(content_img, style_img, num_steps=1000, alpha=8, beta=70, lr=0.004):
    try:
        # Load images
        content = load_image(content_img)
        style = load_image(style_img)
        
        # Initialize target image
        generated_image = content.clone().requires_grad_(True)
        
        # Initialize model
        model = VGG().to(device).eval()
        
        # Setup optimizer
        optimizer = optim.Adam([generated_image], lr=lr)
        
        # Style transfer loop
        for step in range(num_steps):
            # Get features
            gen_features = model(generated_image)
            orig_features = model(content)
            style_features = model(style)
            
            # Calculate loss
            total_loss, content_loss, style_loss = calculate_loss(
                gen_features, orig_features, style_features, alpha, beta
            )
            
            # Optimize
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            if step % 100 == 0:
                print(f"Step {step}: Total Loss: {total_loss.item():.4f}, "
                      f"Content Loss: {content_loss.item():.4f}, "
                      f"Style Loss: {style_loss.item():.4f}")
        
        return generated_image
        
    except Exception as e:
        print(f"Error during style transfer: {str(e)}")
        print(traceback.format_exc())
        raise ValueError(f"Error during style transfer: {str(e)}")

# Global initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

@app.route('/style-transfer', methods=['POST'])
def style_transfer_api():
    try:
        # Validate request
        if 'content' not in request.files or 'style' not in request.files:
            return jsonify({'error': 'Missing content or style image'}), 400
            
        content_image = request.files['content'].read()
        style_image = request.files['style'].read()
        
        if not content_image or not style_image:
            return jsonify({'error': 'Empty image file(s)'}), 400
        
        # Get parameters from request
        alpha = float(request.form.get('alpha', 8))  # Content weight
        beta = float(request.form.get('beta', 70))   # Style weight
        steps = int(request.form.get('steps', 1000))
        learning_rate = float(request.form.get('lr', 0.004))
            
        # Perform style transfer
        styled_image_tensor = style_transfer(
            content_image, 
            style_image,
            num_steps=steps,
            alpha=alpha,
            beta=beta,
            lr=learning_rate
        )
        
        # Convert result to base64
        output_image_base64 = tensor_to_base64(styled_image_tensor)
        
        return jsonify({'output': output_image_base64}), 200
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == "__main__":
    app.run(port=5000, debug=True)