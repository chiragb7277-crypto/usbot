#import <Foundation/Foundation.h>
#import <Vision/Vision.h>
#import <AppKit/AppKit.h>

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        if (argc < 2) return 1;
        NSString *path = [NSString stringWithUTF8String:argv[1]];
        NSURL *url = [NSURL fileURLWithPath:path];
        NSImage *image = [[NSImage alloc] initWithContentsOfURL:url];
        if (!image) return 1;
        CGImageRef cgImage = [image CGImageForProposedRect:NULL context:nil hints:nil];
        if (!cgImage) return 1;

        VNRecognizeTextRequest *request = [[VNRecognizeTextRequest alloc] initWithCompletionHandler:^(VNRequest *req, NSError *error) {
            if (error) return;
            for (VNRecognizedTextObservation *obs in req.results) {
                VNRecognizedText *top = [[obs topCandidates:1] firstObject];
                if (top) {
                    printf("%s\n", [top.string UTF8String]);
                }
            }
        }];
        request.recognitionLevel = VNRequestTextRecognitionLevelAccurate;
        request.usesLanguageCorrection = YES;

        VNImageRequestHandler *handler = [[VNImageRequestHandler alloc] initWithCGImage:cgImage options:@{}];
        NSError *err = nil;
        [handler performRequests:@[request] error:&err];
    }
    return 0;
}
