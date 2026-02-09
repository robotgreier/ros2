#!/usr/bin/env python3
"""
ArUco Marker Generator
Generates ArUco markers for printing and object tagging
"""

import cv2
import numpy as np
import argparse


def generate_aruco_marker(marker_id, marker_size=200, border_bits=1, 
                         dictionary=cv2.aruco.DICT_4X4_50):
    """
    Generate a single ArUco marker
    
    Args:
        marker_id: Marker ID (0-49 for DICT_4X4_50)
        marker_size: Size in pixels (for display/print)
        border_bits: White border size in marker units
        dictionary: ArUco dictionary to use
    
    Returns:
        Marker image (grayscale)
    """
    aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary)
    marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)
    
    # Add white border
    if border_bits > 0:
        border_size = marker_size // 4 * border_bits
        marker_with_border = np.ones(
            (marker_size + 2*border_size, marker_size + 2*border_size),
            dtype=np.uint8
        ) * 255
        marker_with_border[border_size:-border_size, border_size:-border_size] = marker_img
        return marker_with_border
    
    return marker_img


def create_marker_sheet(marker_ids, marker_size=200, markers_per_row=3,
                       dictionary=cv2.aruco.DICT_4X4_50, add_labels=True):
    """
    Create a sheet with multiple markers for printing
    
    Args:
        marker_ids: List of marker IDs to generate
        marker_size: Size of each marker in pixels
        markers_per_row: Number of markers per row
        dictionary: ArUco dictionary
        add_labels: Add ID labels below markers
    
    Returns:
        Sheet image with all markers
    """
    num_markers = len(marker_ids)
    num_rows = (num_markers + markers_per_row - 1) // markers_per_row
    
    # Calculate sheet size
    spacing = 50
    label_height = 40 if add_labels else 0
    cell_width = marker_size + spacing
    cell_height = marker_size + label_height + spacing
    
    sheet_width = markers_per_row * cell_width + spacing
    sheet_height = num_rows * cell_height + spacing
    
    # Create white sheet
    sheet = np.ones((sheet_height, sheet_width), dtype=np.uint8) * 255
    
    # Place markers
    for idx, marker_id in enumerate(marker_ids):
        row = idx // markers_per_row
        col = idx % markers_per_row
        
        # Generate marker
        marker = generate_aruco_marker(marker_id, marker_size, dictionary=dictionary)
        
        # Calculate position
        x = col * cell_width + spacing
        y = row * cell_height + spacing
        
        # Place marker
        sheet[y:y+marker.shape[0], x:x+marker.shape[1]] = marker
        
        # Add label
        if add_labels:
            label_y = y + marker.shape[0] + 25
            label_x = x + marker_size // 2 - 30
            cv2.putText(sheet, f"ID: {marker_id}", (label_x, label_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, 0, 2)
    
    return sheet


def create_object_markers(output_dir='aruco_markers'):
    """
    Create markers for the taskbot project
    Objects (IDs 0-5) and Zones (IDs 10-12)
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # Object type definitions
    objects = {
        0: {'type': 'box_small', 'destination': 'zone_a'},
        1: {'type': 'box_medium', 'destination': 'zone_a'},
        2: {'type': 'cylinder', 'destination': 'zone_b'},
        3: {'type': 'sphere', 'destination': 'zone_b'},
        4: {'type': 'tool', 'destination': 'zone_c'},
        5: {'type': 'tool', 'destination': 'zone_c'},
    }
    
    zones = {
        10: 'zone_a',
        11: 'zone_b',
        12: 'zone_c',
    }
    
    print("Generating ArUco markers for taskbot project...")
    print()
    
    # Generate individual object markers
    print("Object markers:")
    for marker_id, obj_info in objects.items():
        marker = generate_aruco_marker(marker_id, marker_size=400)
        filename = f"{output_dir}/object_{marker_id}_{obj_info['type']}.png"
        cv2.imwrite(filename, marker)
        print(f"  ID {marker_id}: {obj_info['type']:12s} → {obj_info['destination']} "
              f"(saved as {filename})")
    
    print()
    print("Zone markers:")
    for marker_id, zone_name in zones.items():
        marker = generate_aruco_marker(marker_id, marker_size=400)
        filename = f"{output_dir}/zone_{marker_id}_{zone_name}.png"
        cv2.imwrite(filename, marker)
        print(f"  ID {marker_id}: {zone_name:12s} (saved as {filename})")
    
    # Create sheets for printing
    print()
    print("Creating print sheets...")
    
    # Object sheet
    object_ids = list(objects.keys())
    object_sheet = create_marker_sheet(object_ids, marker_size=250, markers_per_row=3)
    object_sheet_file = f"{output_dir}/objects_sheet.png"
    cv2.imwrite(object_sheet_file, object_sheet)
    print(f"  Object sheet: {object_sheet_file}")
    
    # Zone sheet
    zone_ids = list(zones.keys())
    zone_sheet = create_marker_sheet(zone_ids, marker_size=300, markers_per_row=3)
    zone_sheet_file = f"{output_dir}/zones_sheet.png"
    cv2.imwrite(zone_sheet_file, zone_sheet)
    print(f"  Zone sheet: {zone_sheet_file}")
    
    # Complete sheet
    all_ids = object_ids + zone_ids
    complete_sheet = create_marker_sheet(all_ids, marker_size=200, markers_per_row=3)
    complete_sheet_file = f"{output_dir}/complete_sheet.png"
    cv2.imwrite(complete_sheet_file, complete_sheet)
    print(f"  Complete sheet: {complete_sheet_file}")
    
    print()
    print("=" * 60)
    print("PRINTING INSTRUCTIONS:")
    print("=" * 60)
    print("1. Print the sheets at actual size (do not scale)")
    print("2. For best results, print on white paper with good contrast")
    print("3. Recommended marker sizes:")
    print("   - Objects: 50mm x 50mm minimum")
    print("   - Zones: 100mm x 100mm or larger")
    print("4. Attach markers to objects and zone locations")
    print("5. Ensure markers are flat and well-lit during operation")
    print()
    print(f"All files saved to: {output_dir}/")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Generate ArUco markers')
    parser.add_argument('--mode', choices=['project', 'custom', 'single'],
                       default='project',
                       help='Generation mode')
    parser.add_argument('--id', type=int, help='Marker ID for single mode')
    parser.add_argument('--ids', type=int, nargs='+', help='Marker IDs for custom mode')
    parser.add_argument('--size', type=int, default=400, help='Marker size in pixels')
    parser.add_argument('--output', type=str, default='aruco_markers',
                       help='Output directory')
    
    args = parser.parse_args()
    
    if args.mode == 'project':
        create_object_markers(args.output)
    
    elif args.mode == 'single':
        if args.id is None:
            print("Error: --id required for single mode")
            return
        
        print(f"Generating marker ID {args.id}...")
        marker = generate_aruco_marker(args.id, marker_size=args.size)
        filename = f"marker_{args.id}.png"
        cv2.imwrite(filename, marker)
        print(f"Saved: {filename}")
        
        # Also display
        cv2.imshow(f'Marker ID {args.id}', marker)
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    elif args.mode == 'custom':
        if args.ids is None:
            print("Error: --ids required for custom mode")
            return
        
        print(f"Generating {len(args.ids)} markers...")
        sheet = create_marker_sheet(args.ids, marker_size=args.size//2)
        filename = f"custom_sheet.png"
        cv2.imwrite(filename, sheet)
        print(f"Saved: {filename}")
        
        # Also display
        scale = 800 / max(sheet.shape)
        display = cv2.resize(sheet, None, fx=scale, fy=scale)
        cv2.imshow('Custom Sheet', display)
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
