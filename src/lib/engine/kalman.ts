import type { DetectionBox } from '@/types';

/**
 * Simple 2D Kalman filter for face box tracking.
 * State: [x1, y1, x2, y2, dx1, dy1, dx2, dy2]
 * Predicts face position between detections for smooth tracking.
 */
export class KalmanBox {
  // State: position + velocity
  private x1: number; private y1: number;
  private x2: number; private y2: number;
  private dx1: number; private dy1: number;
  private dx2: number; private dy2: number;

  // Process noise (how much we expect the face to accelerate)
  private qPos: number;
  private qVel: number;

  // Measurement noise (how noisy detections are)
  private rPos: number;

  // Uncariance (diagonal)
  private p11: number; private p22: number; private p33: number; private p44: number;
  private p55: number; private p66: number; private p77: number; private p88: number;

  private framesSinceUpdate: number = 0;

  constructor(
    box: DetectionBox,
    qPos: number = 4.0,    // position process noise
    qVel: number = 2.0,    // velocity process noise
    rPos: number = 2.0,    // measurement noise
  ) {
    this.x1 = box.x1; this.y1 = box.y1;
    this.x2 = box.x2; this.y2 = box.y2;
    this.dx1 = 0; this.dy1 = 0;
    this.dx2 = 0; this.dy2 = 0;

    this.qPos = qPos;
    this.qVel = qVel;
    this.rPos = rPos;

    // Initial uncertainty
    this.p11 = 10; this.p22 = 10; this.p33 = 10; this.p44 = 10;
    this.p55 = 100; this.p66 = 100; this.p77 = 100; this.p88 = 100;
  }

  predict(): DetectionBox {
    // Predict state: position += velocity
    this.x1 += this.dx1;
    this.y1 += this.dy1;
    this.x2 += this.dx2;
    this.y2 += this.dy2;

    // Predict uncertainty: P = FPF' + Q
    // Simplified diagonal update
    this.p11 += 2 * this.p55 + this.qPos; this.p55 += this.qVel;
    this.p22 += 2 * this.p66 + this.qPos; this.p66 += this.qVel;
    this.p33 += 2 * this.p77 + this.qPos; this.p77 += this.qVel;
    this.p44 += 2 * this.p88 + this.qPos; this.p88 += this.qVel;

    this.framesSinceUpdate++;

    return { x1: this.x1, y1: this.y1, x2: this.x2, y2: this.y2, confidence: 0.5 };
  }

  update(box: DetectionBox): void {
    // Innovation (residual)
    const y1 = box.x1 - this.x1;
    const y2 = box.y1 - this.y1;
    const y3 = box.x2 - this.x2;
    const y4 = box.y2 - this.y2;

    // Innovation covariance: S = HPH' + R
    const s1 = this.p11 + this.rPos;
    const s2 = this.p22 + this.rPos;
    const s3 = this.p33 + this.rPos;
    const s4 = this.p44 + this.rPos;

    // Kalman gain: K = PH' / S
    const k1 = this.p11 / s1; const k1v = this.p55 / s1;
    const k2 = this.p22 / s2; const k2v = this.p66 / s2;
    const k3 = this.p33 / s3; const k3v = this.p77 / s3;
    const k4 = this.p44 / s4; const k4v = this.p88 / s4;

    // Update state
    this.x1 += k1 * y1; this.dx1 += k1v * y1;
    this.y1 += k2 * y2; this.dy1 += k2v * y2;
    this.x2 += k3 * y3; this.dx2 += k3v * y3;
    this.y2 += k4 * y4; this.dy2 += k4v * y4;

    // Update covariance: P = (I - KH)P
    this.p11 *= (1 - k1); this.p55 *= (1 - k1v);
    this.p22 *= (1 - k2); this.p66 *= (1 - k2v);
    this.p33 *= (1 - k3); this.p77 *= (1 - k3v);
    this.p44 *= (1 - k4); this.p88 *= (1 - k4v);

    this.framesSinceUpdate = 0;
  }

  getState(): DetectionBox {
    return { x1: this.x1, y1: this.y1, x2: this.x2, y2: this.y2, confidence: 0.5 };
  }

  getVelocity(): { dx1: number; dy1: number; dx2: number; dy2: number } {
    return { dx1: this.dx1, dy1: this.dy1, dx2: this.dx2, dy2: this.dy2 };
  }

  getFramesSinceUpdate(): number {
    return this.framesSinceUpdate;
  }
}
