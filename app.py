# app.py - Flask UI for RAG System
import os
import json
import time
from flask import Flask, render_template, request, jsonify, session
from flask_socketio import SocketIO, emit
from datetime import datetime
import threading
import uuid

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch not available - GPU features disabled")

# Import your existing RAG system
from chat10 import FilterExtractionRAG, print_gpu_memory

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global RAG system instance
rag_system = None
system_ready = False

class ChatManager:
    def __init__(self):
        self.conversations = {}  # Store conversation history
        self.active_users = set()
    
    def start_conversation(self, session_id):
        if session_id not in self.conversations:
            self.conversations[session_id] = {
                'id': session_id,
                'start_time': datetime.now().isoformat(),
                'messages': [],
                'query_count': 0
            }
    
    def add_message(self, session_id, role, content, metadata=None):
        if session_id not in self.conversations:
            self.start_conversation(session_id)
        
        message = {
            'id': str(uuid.uuid4()),
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        self.conversations[session_id]['messages'].append(message)
        self.conversations[session_id]['query_count'] = len([
            m for m in self.conversations[session_id]['messages'] 
            if m['role'] == 'user'
        ])
        
        return message
    
    def get_conversation_history(self, session_id, limit=None):
        if session_id not in self.conversations:
            return []
        
        messages = self.conversations[session_id]['messages']
        if limit:
            return messages[-limit:]
        return messages

chat_manager = ChatManager()

def initialize_rag_system():
    """Initialize the RAG system in a background thread"""
    global rag_system, system_ready
    
    print("🚀 Initializing RAG System...")
    socketio.emit('system_status', {'status': 'initializing', 'message': 'Loading AI models...'})
    
    try:
        rag_system = FilterExtractionRAG()
        success = rag_system.initialize_system(use_optimized=True)
        
        if success:
            system_ready = True
            socketio.emit('system_status', {
                'status': 'ready', 
                'message': 'System ready! You can start chatting.',
                'gpu_info': get_gpu_info()
            })
            print("✅ RAG System initialized successfully!")
        else:
            socketio.emit('system_status', {
                'status': 'error', 
                'message': 'Failed to initialize system. Check console for details.'
            })
    
    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        socketio.emit('system_status', {
            'status': 'error', 
            'message': f'Initialization error: {str(e)}'
        })

def get_gpu_info():
    """Get GPU memory information"""
    try:
        if hasattr(torch, 'cuda') and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            return f"GPU: {allocated:.1f}GB / {reserved:.1f}GB"
        return "GPU: Not available"
    except:
        return "GPU: Unknown"

# Routes
@app.route('/')
def index():
    """Main chat interface"""
    return render_template('index.html')

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'ready' if system_ready else 'initializing',
        'system_ready': system_ready,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/chat/history', methods=['GET'])
def get_chat_history():
    """Get chat history for current session"""
    session_id = session.get('session_id', 'default')
    history = chat_manager.get_conversation_history(session_id)
    return jsonify({'history': history})

@app.route('/api/chat/clear', methods=['POST'])
def clear_chat():
    """Clear chat history"""
    session_id = session.get('session_id', 'default')
    chat_manager.conversations.pop(session_id, None)
    return jsonify({'success': True})

@app.route('/api/system/info', methods=['GET'])
def system_info():
    """Get system information"""
    return jsonify({
        'system_ready': system_ready,
        'gpu_info': get_gpu_info(),
        'model_name': getattr(rag_system, 'MODEL_NAME', 'Unknown') if rag_system else 'Unknown',
        'gemini_available': rag_system.gemini_client is not None if rag_system else False
    })

# SocketIO Events
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    session_id = request.sid
    session['session_id'] = session_id
    chat_manager.active_users.add(session_id)
    chat_manager.start_conversation(session_id)
    
    print(f"📱 User connected: {session_id}")
    emit('connected', {
        'session_id': session_id,
        'system_ready': system_ready,
        'message': 'Connected to RAG Chat System'
    })

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnect"""
    session_id = request.sid
    chat_manager.active_users.discard(session_id)
    print(f"📱 User disconnected: {session_id}")

@socketio.on('send_message')
def handle_chat_message(data):
    """Handle incoming chat messages"""
    if not system_ready or rag_system is None:
        emit('receive_message', {
            'role': 'system',
            'content': '⚠️ System is still initializing. Please wait...',
            'timestamp': datetime.now().isoformat(),
            'type': 'error'
        })
        return
    
    session_id = request.sid
    user_message = data.get('message', '').strip()
    
    if not user_message:
        return
    
    # Add user message to chat history
    chat_manager.add_message(session_id, 'user', user_message)
    
    # Send typing indicator
    emit('typing_start', {'message': 'Processing your query...'})
    
    try:
        # Process the query with the RAG system
        start_time = time.time()
        response_data = rag_system.query_to_json(user_message)
        processing_time = time.time() - start_time
        
        # Format the response based on confidence
        confidence = response_data.get('confidence_score', 0)
        
        if confidence >= 0.6:
            # High confidence - show structured JSON response
            formatted_response = format_high_confidence_response(response_data)
            response_type = 'structured'
        elif confidence >= 0.2:
            # Medium confidence - show suggestions
            formatted_response = format_medium_confidence_response(response_data)
            response_type = 'suggestions'
        else:
            # Low confidence
            formatted_response = format_low_confidence_response(response_data)
            response_type = 'fallback'
        
        # Add assistant response to chat history
        metadata = {
            'processing_time': f"{processing_time:.2f}s",
            'confidence_score': confidence,
            'response_type': response_type,
            'raw_data': response_data
        }
        
        chat_manager.add_message(session_id, 'assistant', formatted_response, metadata)
        
        # Send response to client
        emit('receive_message', {
            'role': 'assistant',
            'content': formatted_response,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata,
            'type': response_type
        })
        
        # Send system info update
        emit('system_info_update', {
            'gpu_info': get_gpu_info(),
            'last_processing_time': f"{processing_time:.2f}s"
        })
        
    except Exception as e:
        error_message = f"❌ Error processing your query: {str(e)}"
        print(f"Processing error: {e}")
        
        chat_manager.add_message(session_id, 'assistant', error_message, {'error': True})
        
        emit('receive_message', {
            'role': 'assistant',
            'content': error_message,
            'timestamp': datetime.now().isoformat(),
            'type': 'error'
        })
    
    finally:
        emit('typing_stop')

def format_high_confidence_response(response_data):
    """Format high confidence response with structured JSON"""
    response_parts = ["✅ **I found relevant information for your query!**\n\n"]
    
    # Remove metadata fields for display
    display_data = {k: v for k, v in response_data.items() 
                   if k not in ['user_query', 'confidence_score', 'response_type']}
    
    if display_data:
        response_parts.append("**Extracted Filters:**")
        response_parts.append("```json")
        response_parts.append(json.dumps(display_data, indent=2))
        response_parts.append("```")
    
    # Add confidence info
    confidence = response_data.get('confidence_score', 0)
    response_parts.append(f"\n*Confidence: {confidence:.1%}*")
    
    return "\n".join(response_parts)

def format_medium_confidence_response(response_data):
    """Format medium confidence response with suggestions"""
    response_parts = ["🤔 **I found some related information, but couldn't fully answer your query.**\n\n"]
    
    # Show retrieved filters
    if response_data.get('retrieved_filters'):
        response_parts.append("**Related filters I found:**")
        for i, filter_text in enumerate(response_data['retrieved_filters'][:3], 1):
            response_parts.append(f"{i}. {filter_text}")
        response_parts.append("")
    
    # Show suggestions
    suggestions = response_data.get('suggested_questions')
    if suggestions:
        response_parts.append("**You might want to ask:**")
        response_parts.append(suggestions)
    
    confidence = response_data.get('confidence_score', 0)
    response_parts.append(f"\n*Confidence: {confidence:.1%}*")
    
    return "\n".join(response_parts)

def format_low_confidence_response(response_data):
    """Format low confidence response"""
    response_parts = ["❌ **This question doesn't match my current knowledge base.**\n\n"]
    
    main_response = response_data.get('response', 'I cannot answer this question with the available information.')
    response_parts.append(main_response)
    
    # Offer help
    response_parts.append("\n💡 **Tip:** Try asking about:")
    response_parts.append("- Company filters and categories")
    response_parts.append("- Business locations and types")
    response_parts.append("- Industry classifications")
    response_parts.append("- Employee size ranges")
    
    return "\n".join(response_parts)

# Initialize system when app starts
@app.before_request
def initialize_system():
    """Initialize RAG system when Flask app starts"""
    threading.Thread(target=initialize_rag_system, daemon=True).start()

if __name__ == '__main__':
    print("🌐 Starting Flask RAG Chat Server...")
    print("📍 Access the chat interface at: http://localhost:5000")
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)