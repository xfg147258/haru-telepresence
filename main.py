"""Entry point for the Haru 2.0 telepresence system.

Usage:
    python main.py                    # Standard webcam mode
    python main.py teleconference     # Remote video (screen capture) mode
"""

import sys
import traceback

import cv2
import rclpy


def _select_system(mode: str):
    """Returns an instantiated system object for the requested mode."""
    if mode == 'teleconference':
        from teleconference_system import HaruTeleconferenceSystem
        print('Starting teleconference mode.')
        return HaruTeleconferenceSystem()

    from base_system import HaruIntegratedSystem
    print('Starting webcam mode.')
    return HaruIntegratedSystem()


def main(args=None) -> None:
    rclpy.init(args=args)
    print('=' * 70)
    print('  Haru 2.0 Integrated System')
    print('=' * 70)

    mode = sys.argv[1] if len(sys.argv) > 1 else 'webcam'

    try:
        system = _select_system(mode)
        system.run()
    except KeyboardInterrupt:
        print('\nInterrupted by user.')
    except Exception as exc:  # noqa: BLE001
        print(f'\nFatal error: {exc}')
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()
        rclpy.shutdown()
        print('System shut down.')


if __name__ == '__main__':
    main()
