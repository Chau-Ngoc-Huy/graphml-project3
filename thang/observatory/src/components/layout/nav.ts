import { BarChart3, FlaskConical, GitCompareArrows, Network, Route as RouteIcon, Telescope } from "lucide-react";
import type { Route } from "@/state/AppState";

export const NAV: { id: Route; label: string; hint: string; icon: typeof Telescope }[] = [
  { id: "overview", label: "Tổng quan", hint: "Bài toán, phát hiện chính", icon: Telescope },
  { id: "journey", label: "Hành trình nghiên cứu", hint: "Pipeline từng bước", icon: RouteIcon },
  { id: "playground", label: "Playground", hint: "Chạy truy vấn trên các mô hình", icon: FlaskConical },
  { id: "compare", label: "So sánh mô hình", hint: "Cùng truy vấn, nhiều mô hình", icon: GitCompareArrows },
  { id: "graph", label: "Đồ thị truy vấn", hint: "Câu → khái niệm → ảnh", icon: Network },
  { id: "results", label: "Kết quả & đánh giá", hint: "Số chính thức, ablation", icon: BarChart3 },
];
