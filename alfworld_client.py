#!/usr/bin/env python
"""ALFWorld client for EMAC+ Mode 1 (online DAgger training).

Runs AlfredThorEnv (AI2-THOR visual mode) and sends observations/images
to dagger_server.py via HTTP.

Usage:
    xvfb-run -a python alfworld_client.py [--server-url URL] [--num-rounds N] [--num-envs N]
"""

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import requests

from alfworld.agents.environment.alfred_thor_env import AlfredThorEnv


# Task type prefixes recognized by the server
TASK_PREFIXES = [
    'pick_and_place',
    'pick_clean_then_place',
    'pick_heat_then_place',
    'pick_cool_then_place',
    'look_at_obj',
    'pick_two_obj',
]


def build_config(num_envs, max_steps):
    """Build AlfredThorEnv config dict."""
    alfworld_data = os.environ.get(
        "ALFWORLD_DATA", os.path.expanduser("~/.cache/alfworld")
    )
    return {
        'dataset': {
            'data_path': os.path.join(alfworld_data, 'json_2.1.1', 'train'),
            'eval_id_data_path': os.path.join(alfworld_data, 'json_2.1.1', 'valid_seen'),
            'eval_ood_data_path': os.path.join(alfworld_data, 'json_2.1.1', 'valid_unseen'),
            'num_train_games': num_envs,
            'num_eval_games': -1,
        },
        'env': {
            'task_types': [1, 2, 3, 4, 5, 6],
            'goal_desc_human_anns_prob': 1.0,
            'thor': {
                'screen_height': 300,
                'screen_width': 300,
                'smooth_nav': False,
                'save_frames_to_disk': False,
                'save_frames_path': '',
            },
        },
        'controller': {
            'type': 'oracle',
            'load_receps': False,
            'debug': False,
        },
        'general': {
            'training_method': 'dagger',
        },
        'dagger': {
            'training': {
                'max_nb_steps_per_episode': max_steps,
            },
        },
    }


def extract_task_type(infos):
    """Extract task type string from the game file path in infos.

    Returns a string like 'pick_and_place_simple-Pencil-Shelf-226' that
    starts with one of the recognized TASK_PREFIXES.
    """
    gamefile_list = infos.get('extra.gamefile', [''])
    gamefile = gamefile_list[0] if isinstance(gamefile_list, list) else gamefile_list
    if not gamefile:
        return ''
    parts = gamefile.replace('\\', '/').split('/')
    for part in parts:
        for prefix in TASK_PREFIXES:
            if part.startswith(prefix):
                return part
    return ''


def send_to_server(server_url, obs, history, infos, task_type, done, image):
    """Send observation data to dagger_server and return the action string.

    Protocol (matching dagger_server.py process_feedback):
        POST body: [#OBSERVATION]{obs}[#HISTORY]{history}[#INFORMATION]{info}
                    [#TYPE]{task_type}[#DONE]{True/False}[#IMAGE]{image_json}
        Headers: Content-Length, MD5
        Response: action string with MD5 header for verification
    """
    # Build info dict
    admissible = infos.get('admissible_commands', [[]])
    if isinstance(admissible, list) and len(admissible) > 0:
        admissible = admissible[0] if isinstance(admissible[0], list) else admissible
    info_dict = {'admissible_commands': admissible}

    # JSON-encode image (numpy uint8 array → nested list)
    image_json = json.dumps(image.tolist())

    # Build body matching the server's regex pattern
    body = (
        f"[#OBSERVATION]{obs}"
        f"[#HISTORY]{history}"
        f"[#INFORMATION]{info_dict}"
        f"[#TYPE]{task_type}"
        f"[#DONE]{done}"
        f"[#IMAGE]{image_json}"
    )

    body_bytes = body.encode('utf-8')
    md5 = hashlib.md5(body_bytes).hexdigest()

    headers = {
        'Content-Length': str(len(body_bytes)),
        'MD5': md5,
    }

    response = requests.post(server_url, data=body_bytes, headers=headers)
    response.raise_for_status()

    # Verify response integrity
    response_md5 = response.headers.get('MD5')
    calculated_md5 = hashlib.md5(response.content).hexdigest()
    if response_md5 != calculated_md5:
        print(f"WARNING: Response MD5 mismatch (expected {response_md5}, got {calculated_md5})")

    return response.text


def main():
    parser = argparse.ArgumentParser(
        description='ALFWorld Thor client for EMAC+ DAgger training'
    )
    parser.add_argument(
        '--server-url', type=str, default='http://localhost:7860',
        help='DAgger server URL (default: http://localhost:7860)'
    )
    parser.add_argument(
        '--num-rounds', type=int, default=2,
        help='Number of DAgger rounds (must match server num_rounds)'
    )
    parser.add_argument(
        '--num-envs', type=int, default=3,
        help='Number of environments per round (must match server num_envs)'
    )
    parser.add_argument(
        '--max-steps', type=int, default=50,
        help='Max steps per episode'
    )
    args = parser.parse_args()

    server_url = args.server_url
    num_rounds = args.num_rounds
    num_envs = args.num_envs
    max_steps = args.max_steps

    # Build config and initialize environment
    config = build_config(num_envs, max_steps)

    print(f"Initializing AlfredThorEnv (AI2-THOR visual mode)...")
    print(f"  num_envs={num_envs}, num_rounds={num_rounds}, max_steps={max_steps}")
    env = AlfredThorEnv(config, train_eval='train')
    env.init_env(batch_size=1)

    print(f"Server: {server_url}")
    print(f"Starting DAgger client loop...\n")

    for round_idx in range(num_rounds):
        print(f"{'='*60}")
        print(f"Round {round_idx + 1}/{num_rounds}")
        print(f"{'='*60}")

        for env_idx in range(num_envs):
            # Reset environment — picks a random game file, resets AI2-THOR scene
            obs_list, infos = env.reset()
            obs = obs_list[0]

            # Get image from AI2-THOR (batch, 300, 300, 3)
            frames = env.get_frames()
            image = frames[0]

            # Extract task type from game file path
            task_type = extract_task_type(infos)

            # Initial history is just the observation
            history = obs

            print(f"\n  Env {env_idx + 1}/{num_envs}: {task_type}")
            print(f"    Init obs: {obs[:120]}...")

            # Send initial observation with task_type
            action_str = send_to_server(
                server_url, obs, history, infos, task_type, False, image
            )

            # Check for SKIP (server sends "['SKIP']" for already-solved envs)
            if "SKIP" in action_str:
                print(f"    SKIPPED (already solved)")
                continue

            # Parse action from response (server sends str(list), e.g. "['go to desk 1']")
            action = action_str.strip("[] '\"")
            print(f"    Step 0: {action}")

            done = False
            for step in range(1, max_steps + 1):
                # Execute action in AI2-THOR environment
                obs_list, _, dones, infos = env.step([action])
                obs = obs_list[0]

                # Check if task was won
                won_list = infos.get('won', [False])
                won = won_list[0] if isinstance(won_list, list) else won_list
                if won:
                    done = True

                # Get new frame from AI2-THOR
                frames = env.get_frames()
                image = frames[0]

                # Append to history
                history = history + f"\n> {action}\n{obs}"

                if done:
                    print(f"    DONE at step {step} (won={won})")
                    # Send final observation with done=True
                    send_to_server(
                        server_url, obs, history, infos, "", True, image
                    )
                    break

                # Send observation and get next action
                action_str = send_to_server(
                    server_url, obs, history, infos, "", False, image
                )
                action = action_str.strip("[] '\"")
                print(f"    Step {step}: {action}")

            if not done:
                print(f"    MAX STEPS reached ({max_steps})")

        print(f"\n  Round {round_idx + 1} complete.")

    # Send one final message to unblock the server's transition for the last env.
    # The server waits at feedback_queue.get() after processing each action;
    # this final reset+send triggers the transition that exits the inner loop,
    # allowing the server to run update_memory, save checkpoints, and finish.
    print(f"\nSending final transition message...")
    obs_list, infos = env.reset()
    obs = obs_list[0]
    frames = env.get_frames()
    image = frames[0]
    task_type = extract_task_type(infos)
    send_to_server(server_url, obs, obs, infos, task_type, False, image)

    print("All rounds complete. Client exiting.")


if __name__ == "__main__":
    main()
