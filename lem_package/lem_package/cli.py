"""
Command-line interface for the Large Event Model (LEM).
"""

import argparse
import json
import os
from datetime import datetime
from .inference import load_resources, preprocess_event, get_event_embedding, suggest_device_action

def main():
    """
    Main entry point for the LEM CLI.
    """
    parser = argparse.ArgumentParser(description='Large Event Model (LEM) CLI')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Preprocess command
    preprocess_parser = subparsers.add_parser('preprocess', help='Preprocess an event')
    preprocess_parser.add_argument('event_json', help='JSON string or file path containing the event')
    
    # Embed command
    embed_parser = subparsers.add_parser('embed', help='Generate an embedding for a sequence of events')
    embed_parser.add_argument('events_json', help='JSON string or file path containing the events')
    embed_parser.add_argument('--output', '-o', help='Output file path for the embedding')
    
    # Suggest command
    suggest_parser = subparsers.add_parser('suggest', help='Suggest a device action based on recent events')
    suggest_parser.add_argument('events_json', help='JSON string or file path containing the recent events')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Load resources
    model, device_encoder, capability_encoder, state_encoder, historical_embeddings, historical_actions = load_resources()
    
    if args.command == 'preprocess':
        # Load event
        if os.path.isfile(args.event_json):
            with open(args.event_json, 'r') as f:
                event = json.load(f)
        else:
            event = json.loads(args.event_json)
        
        # Preprocess event
        features = preprocess_event(event, device_encoder, capability_encoder, state_encoder)
        
        # Print features
        print(f"Preprocessed features: {features}")
    
    elif args.command == 'embed':
        # Load events
        if os.path.isfile(args.events_json):
            with open(args.events_json, 'r') as f:
                events = json.load(f)
        else:
            events = json.loads(args.events_json)
        
        # Generate embedding
        embedding = get_event_embedding(events, model)
        
        # Save or print embedding
        if args.output:
            import torch
            torch.save(embedding, args.output)
            print(f"Embedding saved to {args.output}")
        else:
            print(f"Embedding: {embedding}")
    
    elif args.command == 'suggest':
        # Load events
        if os.path.isfile(args.events_json):
            with open(args.events_json, 'r') as f:
                events = json.load(f)
        else:
            events = json.loads(args.events_json)
        
        # Suggest action
        action = suggest_device_action(events, historical_embeddings, historical_actions, model)
        
        # Print action
        print(f"Suggested action: {json.dumps(action, indent=2)}")
    
    else:
        parser.print_help()

if __name__ == '__main__':
    main() 