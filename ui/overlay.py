from __future__ import annotations

import argparse
import sys


def main() -> None:
    p = argparse.ArgumentParser(description="Mac overlay window hidden from screen share")
    p.add_argument("--url", required=True, help="http://127.0.0.1:8765/")
    p.add_argument("--width", type=int, default=420)
    p.add_argument("--height", type=int, default=640)
    args = p.parse_args()

    try:
        from Cocoa import (
            NSApplication,
            NSColor,
            NSFloatingWindowLevel,
            NSMakeRect,
            NSObject,
            NSScreen,
            NSWindow,
            NSWindowCollectionBehaviorCanJoinAllSpaces,
            NSWindowCollectionBehaviorFullScreenAuxiliary,
            NSWindowStyleMaskClosable,
            NSWindowStyleMaskFullSizeContentView,
            NSWindowStyleMaskResizable,
            NSWindowStyleMaskTitled,
        )
        from Foundation import NSURL, NSURLRequest
        from WebKit import WKWebView, WKWebViewConfiguration
    except ImportError:
        print("overlay needs pyobjc: pip install pyobjc-framework-Cocoa pyobjc-framework-WebKit", file=sys.stderr)
        raise SystemExit(1)

    # NSWindowSharingNone — excluded from Zoom/Meet screen share and screenshots.
    NSWindowSharingNone = 0

    class Delegate(NSObject):
        def applicationShouldTerminateAfterLastWindowClosed_(self, app):
            return True

    app = NSApplication.sharedApplication()
    app.setActivationPolicy_(1)
    delegate = Delegate.alloc().init()
    app.setDelegate_(delegate)

    screen = NSScreen.mainScreen().visibleFrame()
    x = screen.origin.x + screen.size.width - args.width - 18
    y = screen.origin.y + screen.size.height - args.height - 18
    style = (
        NSWindowStyleMaskTitled
        | NSWindowStyleMaskClosable
        | NSWindowStyleMaskResizable
        | NSWindowStyleMaskFullSizeContentView
    )
    win = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
        NSMakeRect(x, y, args.width, args.height), style, 2, False
    )
    win.setTitle_("Copilot")
    win.setTitlebarAppearsTransparent_(True)
    win.setBackgroundColor_(NSColor.clearColor())
    win.setOpaque_(False)
    win.setLevel_(NSFloatingWindowLevel)
    win.setSharingType_(NSWindowSharingNone)
    win.setCollectionBehavior_(
        NSWindowCollectionBehaviorCanJoinAllSpaces | NSWindowCollectionBehaviorFullScreenAuxiliary
    )
    win.setHasShadow_(True)

    cfg = WKWebViewConfiguration.alloc().init()
    view = WKWebView.alloc().initWithFrame_configuration_(win.contentView().bounds(), cfg)
    view.setAutoresizingMask_(18)
    view.setValue_forKey_(False, "drawsBackground")
    req = NSURLRequest.requestWithURL_(NSURL.URLWithString_(args.url))
    view.loadRequest_(req)
    win.setContentView_(view)
    win.makeKeyAndOrderFront_(None)
    app.activateIgnoringOtherApps_(False)
    app.run()


if __name__ == "__main__":
    main()
