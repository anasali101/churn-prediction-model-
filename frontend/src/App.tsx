import jsPDF from "jspdf";
import autoTable from "jspdf-autotable";
import { useEffect, useState } from "react";
import axios from "axios";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
} from "recharts";
import {
  Activity,
  Users,
  TrendingUp,
  ShieldCheck,
  RefreshCw,
  AlertTriangle,
  CheckCircle2,
  Lightbulb,
  Wifi,
  WifiOff,
} from "lucide-react";
import "./App.css";

const API_URL =
  import.meta.env.VITE_API_URL || "http://localhost:8000";

type RiskSegment = "High" | "Medium" | "Low";

interface CustomerInput {
  tenure: number;
  monthly_charges: number;
  contract: string;
}

interface ShapContribution {
  feature: string;
  value: number | string;
  impact: number;
}

interface Prediction {
  churn_risk_percent: number;
  risk_segment: RiskSegment;
  recommendation: string;
  model_type: string;
  explanation_method: string[];
  shap_explanation: ShapContribution[];
  shap_base_value: number;
  shap_output_value: number;
  shap_output_scale: string;
  lime_explanation?: {
    feature_condition: string;
    impact: number;
  }[];
  customer_features: Record<string, number>;
}

interface HistoryItem {
  id: number;
  date: string;
  customer: CustomerInput;
  risk: number;
  segment: RiskSegment;
}

interface BatchCustomer {
  customer_index: number;
  churn_risk_percent: number;
  risk_segment: RiskSegment;
}

interface CsvBatchResult {
  total_customers: number;
  high_risk: number;
  medium_risk: number;
  low_risk: number;
  customers: BatchCustomer[];
}

interface EvaluationResult {
  evaluation_type: string;
  test_customers: number;
  roc_auc: number;
  accuracy: number;
  churn_rate_percent: number;
  roc_curve?: { fpr: number; tpr: number }[];
  confusion_matrix?: {
    true_negative: number;
    false_positive: number;
    false_negative: number;
    true_positive: number;
    labels?: string[];
  };
  note?: string;
}

interface RetentionRecommendation {
  action: string;
  reason: string;
  priority: string;
}

interface RetentionResult {
  risk_segment?: string;
  churn_probability_percent?: number;
  priority?: string;
  recommendations?: RetentionRecommendation[];
  disclaimer?: string;
}

// Custom type to safely access injected jspdf-autotable properties
type jsPDFWithAutoTable = jsPDF & { lastAutoTable?: { finalY: number } };

const initialCustomer: CustomerInput = {
  tenure: 2,
  monthly_charges: 89,
  contract: "Monthly",
};

function ShapWaterfall({
  baseValue,
  contributions,
  outputValue,
}: {
  baseValue: number;
  contributions: ShapContribution[];
  outputValue: number;
}) {
  const items = [...contributions].sort(
    (a, b) => Math.abs(b.impact) - Math.abs(a.impact)
  );

  let cumulative = baseValue;

  const bars = items.map((item) => {
    const start = cumulative;
    const end = start + item.impact;
    cumulative = end;

    return { ...item, start, end };
  });

  const values = [
    baseValue,
    outputValue,
    ...bars.flatMap((bar) => [bar.start, bar.end]),
  ];

  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = Math.max(max - min, 0.1);
  const padding = range * 0.12;

  const domainMin = min - padding;
  const domainMax = max + padding;

  const left = 250;
  const plotWidth = 650;
  const rowHeight = 36;
  const top = 44;
  const height = top + bars.length * rowHeight + 30;

  const x = (value: number) =>
    left + ((value - domainMin) / (domainMax - domainMin)) * plotWidth;

  const zeroX = x(0);

  return (
    <div className="shap-waterfall">
      <div className="shap-summary">
        <div>
          <span>Baseline output</span>
          <strong>{baseValue.toFixed(4)}</strong>
        </div>
        <div>
          <span>Final model output</span>
          <strong>{outputValue.toFixed(4)}</strong>
        </div>
      </div>

      <div className="shap-chart-scroll">
        <svg
          viewBox={`0 0 960 ${height}`}
          width="100%"
          role="img"
          aria-label="SHAP waterfall chart showing each feature's contribution to the model output"
        >
          <text x={left} y={20} fontSize="12" fill="currentColor">
            Model output (raw margin / log-odds)
          </text>

          <line
            x1={zeroX}
            x2={zeroX}
            y1={top - 8}
            y2={height - 22}
            stroke="#94a3b8"
            strokeDasharray="4 4"
          />

          {bars.map((bar, index) => {
            const y = top + index * rowHeight;
            const barX = Math.min(x(bar.start), x(bar.end));
            const barWidth = Math.max(
              Math.abs(x(bar.end) - x(bar.start)),
              1
            );

            const color = bar.impact >= 0 ? "#dc684b" : "#3984d6";

            return (
              <g key={bar.feature}>
                {index > 0 && (
                  <line
                    x1={x(bars[index - 1].end)}
                    x2={x(bars[index - 1].end)}
                    y1={y - rowHeight + 5}
                    y2={y + 5}
                    stroke="#94a3b8"
                    strokeDasharray="3 3"
                  />
                )}

                <text
                  x={left - 12}
                  y={y + 15}
                  textAnchor="end"
                  fontSize="12"
                  fill="currentColor"
                >
                  {bar.feature}
                </text>

                <rect
                  x={barX}
                  y={y}
                  width={barWidth}
                  height={20}
                  rx={3}
                  fill={color}
                >
                  <title>
                    {`${bar.feature}: contribution ${bar.impact.toFixed(
                      5
                    )}, customer value ${bar.value}`}
                  </title>
                </rect>

                <text
                  x={Math.min(
                    left + plotWidth + 8,
                    barX + barWidth + 6
                  )}
                  y={y + 14}
                  fontSize="11"
                  fill="currentColor"
                >
                  {bar.impact >= 0 ? "+" : ""}
                  {bar.impact.toFixed(4)}
                </text>
              </g>
            );
          })}

          <text x={left} y={height - 5} fontSize="11" fill="currentColor">
            Each bar moves the previous output toward the next output.
          </text>
        </svg>
      </div>

      <div className="shap-legend">
        <span>
          <i className="shap-dot positive" />
          Positive contribution
        </span>
        <span>
          <i className="shap-dot negative" />
          Negative contribution
        </span>
      </div>
    </div>
  );
}

function App() {
  const [retentionResult, setRetentionResult] = useState<RetentionResult | null>(null);
  const [retentionLoading, setRetentionLoading] = useState(false);
  const [retentionError, setRetentionError] = useState("");

  const [customer, setCustomer] = useState<CustomerInput>(initialCustomer);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>([]);

  const [apiOnline, setApiOnline] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [batchResult, setBatchResult] = useState<CsvBatchResult | null>(null);
  const [selectedCsv, setSelectedCsv] = useState<File | null>(null);
  const [batchLoading, setBatchLoading] = useState(false);
  const [batchError, setBatchError] = useState("");

  const [rocMetrics, setRocMetrics] = useState<EvaluationResult | null>(null);
  const [metricsLoading, setMetricsLoading] = useState(false);
  const [metricsError, setMetricsError] = useState("");

  useEffect(() => {
    let active = true;

    const checkApi = async () => {
      try {
        await axios.get(`${API_URL}/health`, { timeout: 3000 });
        if (active) setApiOnline(true);
      } catch {
        if (active) setApiOnline(false);
      }
    };

    void checkApi();
    const timer = window.setInterval(checkApi, 10000);

    return () => {
      active = false;
      window.clearInterval(timer);
    };
  }, []);

  const loadEvaluation = async () => {
    setMetricsLoading(true);
    setMetricsError("");

    try {
      const response = await axios.get<EvaluationResult>(`${API_URL}/evaluate`, {
        timeout: 120000,
      });
      setRocMetrics(response.data);
    } catch (err) {
      if (axios.isAxiosError(err)) {
        setMetricsError(
          (err.response?.data as { detail?: string })?.detail ||
          err.message ||
          "Could not load model evaluation."
        );
      } else {
        setMetricsError("An unexpected error occurred while loading evaluation metrics.");
      }
    } finally {
      setMetricsLoading(false);
    }
  };

  const predictChurn = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setLoading(true);
    setError("");
    setPrediction(null);
    setRetentionResult(null);

    try {
      const response = await axios.post<Prediction>(
        `${API_URL}/predict`,
        {
          tenure: customer.tenure,
          monthly_charges: customer.monthly_charges,
          contract: customer.contract,
        },
        { timeout: 15000 }
      );

      const result = response.data;
      setPrediction(result);

      setHistory((previous) =>
        [
          {
            id: Date.now(),
            date: new Date().toLocaleTimeString(),
            customer: { ...customer },
            risk: result.churn_risk_percent,
            segment: result.risk_segment,
          },
          ...previous,
        ].slice(0, 8)
      );
    } catch (err) {
      if (axios.isAxiosError(err)) {
        setError(
          (err.response?.data as { detail?: string })?.detail ||
          "Cannot connect to the API. Check that FastAPI is running on port 8000."
        );
      } else {
        setError("An unexpected error occurred.");
      }
    } finally {
      setLoading(false);
    }
  };

  const loadRetentionRecommendations = async () => {
    if (!prediction) {
      setRetentionError("Predict a customer first.");
      return;
    }

    setRetentionLoading(true);
    setRetentionError("");
    setRetentionResult(null);

    try {
      const response = await axios.post<RetentionResult>(
        `${API_URL}/retention-recommendations`,
        {
          customer: {
            tenure: customer.tenure,
            Contract:
              customer.contract === "Monthly"
                ? "Month-to-month"
                : customer.contract,
            monthly_charges: customer.monthly_charges,
          },
          churn_probability: prediction.churn_risk_percent / 100,
        },
        { timeout: 15000 }
      );

      setRetentionResult(response.data);
    } catch (err) {
      if (axios.isAxiosError(err)) {
        setRetentionError(
          (err.response?.data as { detail?: string })?.detail ||
            err.message ||
            "Could not load retention recommendations. Check that the backend endpoint is running."
        );
      } else {
        setRetentionError("An unexpected error occurred while generating recommendations.");
      }
    } finally {
      setRetentionLoading(false);
    }
  };

  const downloadPredictionPdf = () => {
    if (!prediction) {
      setError("Please predict a customer before downloading the PDF.");
      return;
    }

    const doc = new jsPDF();

    doc.setFontSize(20);
    doc.text("ChurnIQ - Customer Churn Report", 14, 20);

    doc.setFontSize(11);
    doc.text(`Generated: ${new Date().toLocaleString()}`, 14, 30);
    doc.text(`Model: ${prediction.model_type}`, 14, 38);

    autoTable(doc, {
      startY: 48,
      head: [["Metric", "Value"]],
      body: [
        ["Customer tenure", `${customer.tenure} months`],
        ["Monthly charges", customer.monthly_charges.toFixed(2)],
        ["Contract", customer.contract],
        ["Churn risk", `${prediction.churn_risk_percent}%`],
        ["Risk segment", prediction.risk_segment],
        ["Retention recommendation", prediction.recommendation],
      ],
      styles: { overflow: "linebreak", cellPadding: 3 },
      columnStyles: { 0: { cellWidth: 55 }, 1: { cellWidth: 125 } },
    });

    if (prediction.shap_explanation && prediction.shap_explanation.length > 0) {
      const finalY = (doc as jsPDFWithAutoTable).lastAutoTable?.finalY ?? 80;
      autoTable(doc, {
        startY: finalY + 10,
        head: [["Feature", "Customer value", "SHAP impact"]],
        body: prediction.shap_explanation.map((item) => [
          item.feature,
          String(item.value),
          item.impact.toFixed(5),
        ]),
      });
    }

    doc.save("ChurnIQ-Customer-Report.pdf");
  };

  const uploadCsv = async () => {
    if (!selectedCsv) {
      setBatchError("Please choose a CSV file first.");
      return;
    }
    if (!selectedCsv.name.toLowerCase().endsWith(".csv")) {
      setBatchError("Please select a .csv file.");
      return;
    }

    setBatchLoading(true);
    setBatchError("");
    setBatchResult(null);

    try {
      const formData = new FormData();
      formData.append("file", selectedCsv);
      const response = await axios.post<CsvBatchResult>(
        `${API_URL}/predict-csv`,
        formData,
        { timeout: 120000 }
      );
      setBatchResult(response.data);
    } catch (err) {
      if (axios.isAxiosError(err)) {
        setBatchError(
          (err.response?.data as { detail?: string })?.detail ||
          err.message ||
          "CSV upload failed. Check that the backend has a /predict-csv endpoint."
        );
      } else {
        setBatchError("An unexpected error occurred while processing the CSV.");
      }
    } finally {
      setBatchLoading(false);
    }
  };

  const downloadBatchPdf = () => {
    if (!batchResult) return;
    const doc = new jsPDF();

    doc.setFontSize(20);
    doc.text("ChurnIQ - Batch Churn Report", 14, 20);

    doc.setFontSize(10);
    doc.text(`Generated: ${new Date().toLocaleString()}`, 14, 29);
    doc.text(`Customers analyzed: ${batchResult.total_customers}`, 14, 36);
    doc.text(`High risk: ${batchResult.high_risk}`, 14, 43);
    doc.text(`Medium risk: ${batchResult.medium_risk}`, 14, 50);
    doc.text(`Low risk: ${batchResult.low_risk}`, 14, 57);

    autoTable(doc, {
      startY: 65,
      head: [["CSV row", "Churn risk (%)", "Risk segment"]],
      body: (batchResult.customers || []).map((item) => [
        String(item.customer_index),
        `${Number(item.churn_risk_percent).toFixed(2)}%`,
        item.risk_segment,
      ]),
      styles: { cellPadding: 2 },
    });

    doc.save("ChurnIQ-Batch-Report.pdf");
  };

  const downloadEvaluationPdf = () => {
    if (!rocMetrics) {
      setMetricsError("Load model evaluation before downloading its PDF.");
      return;
    }

    const doc = new jsPDF();
    const pageWidth = doc.internal.pageSize.getWidth();

    doc.setFontSize(19);
    doc.text("ChurnIQ - Model Evaluation Report", 14, 19);
    doc.setFontSize(9);
    doc.text(`Generated: ${new Date().toLocaleString()}`, 14, 27);

    autoTable(doc, {
      startY: 33,
      head: [["Evaluation metric", "Value"]],
      body: [
        ["Evaluation type", String(rocMetrics.evaluation_type ?? "Not specified")],
        ["Test customers", String(rocMetrics.test_customers ?? "—")],
        ["ROC-AUC", typeof rocMetrics.roc_auc === "number" ? rocMetrics.roc_auc.toFixed(4) : "—"],
        ["Accuracy", typeof rocMetrics.accuracy === "number"
          ? `${(rocMetrics.accuracy <= 1 ? rocMetrics.accuracy * 100 : rocMetrics.accuracy).toFixed(2)}%`
          : "—"],
        ["Actual churn rate", typeof rocMetrics.churn_rate_percent === "number"
          ? `${rocMetrics.churn_rate_percent.toFixed(2)}%`
          : "—"],
      ],
      styles: { fontSize: 9, cellPadding: 2.5 },
      margin: { left: 14, right: 14 },
    });

    let y = ((doc as jsPDFWithAutoTable).lastAutoTable?.finalY ?? 80) + 12;
    doc.setFontSize(13);
    doc.text("ROC Curve", 14, y);
    y += 7;

    const chartLeft = 27;
    const chartTop = y + 2;
    const chartWidth = 150;
    const chartHeight = 66;
    const chartBottom = chartTop + chartHeight;

    doc.setDrawColor(100, 116, 139);
    doc.setLineWidth(0.35);
    doc.line(chartLeft, chartTop, chartLeft, chartBottom);
    doc.line(chartLeft, chartBottom, chartLeft + chartWidth, chartBottom);
    doc.setDrawColor(148, 163, 184);
    doc.setLineDashPattern([2, 2], 0);
    doc.line(chartLeft, chartBottom, chartLeft + chartWidth, chartTop);
    doc.setLineDashPattern([], 0);

    const points = rocMetrics.roc_curve || [];
    if (points.length > 0) {
      doc.setDrawColor(37, 99, 235);
      doc.setLineWidth(0.8);
      for (let i = 1; i < points.length; i++) {
        const p0 = points[i - 1];
        const p1 = points[i];
        const x0 = chartLeft + Math.max(0, Math.min(1, Number(p0.fpr))) * chartWidth;
        const y0 = chartBottom - Math.max(0, Math.min(1, Number(p0.tpr))) * chartHeight;
        const x1 = chartLeft + Math.max(0, Math.min(1, Number(p1.fpr))) * chartWidth;
        const y1 = chartBottom - Math.max(0, Math.min(1, Number(p1.tpr))) * chartHeight;
        doc.line(x0, y0, x1, y1);
      }
    } else {
      doc.setFontSize(9);
      doc.text("ROC curve points are not available in the /evaluate response.", 14, chartBottom + 8);
    }

    doc.setFontSize(8);
    doc.text("0", chartLeft - 2, chartBottom + 5);
    doc.text("1", chartLeft + chartWidth - 1, chartBottom + 5);
    doc.text("1", chartLeft - 5, chartTop + 2);
    doc.text("False-positive rate", chartLeft + 55, chartBottom + 11);
    doc.text("True-positive rate", 14, chartTop + chartHeight / 2, { angle: 90 });
    doc.text(`AUC: ${typeof rocMetrics.roc_auc === "number" ? rocMetrics.roc_auc.toFixed(4) : "N/A"}`, chartLeft + 4, chartTop + 7);

    y = chartBottom + 23;
    const cm = rocMetrics.confusion_matrix;
    doc.setFontSize(13);
    doc.text("Confusion Matrix", 14, y);
    y += 5;

    if (cm) {
      const x0 = 42;
      const cellW = 42;
      const cellH = 15;
      doc.setFontSize(8);
      doc.text("Predicted: No churn", x0, y + 5);
      doc.text("Predicted: Churn", x0 + cellW, y + 5);
      doc.text("Actual: No churn", 14, y + 14);
      doc.text("Actual: Churn", 14, y + 30);

      const cells = [
        { x: x0, yy: y + 7, value: cm.true_negative, label: "True negative", fill: [220, 252, 231] as [number, number, number] },
        { x: x0 + cellW, yy: y + 7, value: cm.false_positive, label: "False positive", fill: [254, 226, 226] as [number, number, number] },
        { x: x0, yy: y + 23, value: cm.false_negative, label: "False negative", fill: [254, 226, 226] as [number, number, number] },
        { x: x0 + cellW, yy: y + 23, value: cm.true_positive, label: "True positive", fill: [220, 252, 231] as [number, number, number] },
      ];

      cells.forEach((cell) => {
        doc.setFillColor(cell.fill[0], cell.fill[1], cell.fill[2]);
        doc.setDrawColor(203, 213, 225);
        doc.rect(cell.x, cell.yy, cellW, cellH, "FD");
        doc.setTextColor(15, 23, 42);
        doc.setFontSize(11);
        doc.text(String(cell.value ?? "—"), cell.x + cellW / 2, cell.yy + 6, { align: "center" });
        doc.setFontSize(7);
        doc.text(cell.label, cell.x + cellW / 2, cell.yy + 11, { align: "center" });
      });
      doc.setTextColor(0, 0, 0);
    } else {
      doc.setFontSize(9);
      doc.text("Confusion matrix data is not available in the /evaluate response.", 14, y + 8);
    }

    doc.addPage();
    doc.setFontSize(16);
    doc.text("Evaluation Raw Output", 14, 19);
    doc.setFontSize(9);
    doc.text("Confusion matrix counts and ROC curve coordinates returned by the backend.", 14, 27);

    if (cm) {
      autoTable(doc, {
        startY: 34,
        head: [["Confusion matrix value", "Count"]],
        body: [
          ["True negative", String(cm.true_negative)],
          ["False positive", String(cm.false_positive)],
          ["False negative", String(cm.false_negative)],
          ["True positive", String(cm.true_positive)],
        ],
        styles: { fontSize: 9, cellPadding: 2.5 },
      });
    }

    const rocRows = points.map((point, index) => [
      String(index + 1),
      Number(point.fpr).toFixed(6),
      Number(point.tpr).toFixed(6),
    ]);

    autoTable(doc, {
      startY: ((doc as jsPDFWithAutoTable).lastAutoTable?.finalY ?? 35) + 10,
      head: [["Point", "False-positive rate (FPR)", "True-positive rate (TPR)"]],
      body: rocRows.length ? rocRows : [["—", "ROC data unavailable", "ROC data unavailable"]],
      styles: { fontSize: 8, cellPadding: 2 },
      margin: { left: 14, right: 14 },
    });

    if (rocMetrics.note) {
      const noteY = ((doc as jsPDFWithAutoTable).lastAutoTable?.finalY ?? 45) + 8;
      const noteLines = doc.splitTextToSize(`Note: ${rocMetrics.note}`, pageWidth - 28);

      if (noteY + noteLines.length * 4 > 285) {
        doc.addPage();
        doc.text(noteLines, 14, 20);
      } else {
        doc.text(noteLines, 14, noteY);
      }
    }

    doc.save("ChurnIQ-Model-Evaluation-Report.pdf");
  };

  const riskClass = (segment: string) => segment.toLowerCase();

  const chartData = history
    .map((item, index) => ({
      name: `#${history.length - index}`,
      risk: item.risk,
    }))
    .reverse();

  const highRiskCount = history.filter((item) => item.segment === "High").length;

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">
            <Activity size={25} />
          </div>
          <div>
            <h2>ChurnIQ</h2>
            <span>Customer Intelligence</span>
          </div>
        </div>

        <div
  className="nav-item active"
  role="button"
  tabIndex={0}
  onClick={() => {
    document
      .querySelector(".topbar")
      ?.scrollIntoView({ behavior: "smooth", block: "start" });
  }}
  onKeyDown={(event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      document
        .querySelector(".topbar")
        ?.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }}
>
  <Activity size={18} />
  Dashboard
</div>
        <div
          className="nav-item"
          role="button"
          tabIndex={0}
          onClick={() =>
            document
              .getElementById("customer-analysis")
              ?.scrollIntoView({ behavior: "smooth", block: "start" })
          }
          onKeyDown={(event) => {
            if (event.key === "Enter" || event.key === " ") {
              event.preventDefault();
              document
                .getElementById("customer-analysis")
                ?.scrollIntoView({ behavior: "smooth", block: "start" });
            }
          }}
        >
          <Users size={18} />
          Customer analysis
        </div>
        <div
          className="nav-item"
          role="button"
          tabIndex={0}
          onClick={() =>
            document
              .getElementById("retention-insights")
              ?.scrollIntoView({ behavior: "smooth", block: "start" })
          }
          onKeyDown={(event) => {
            if (event.key === "Enter" || event.key === " ") {
              event.preventDefault();
              document
                .getElementById("retention-insights")
                ?.scrollIntoView({ behavior: "smooth", block: "start" });
            }
          }}
        >
          <Lightbulb size={18} />
          Retention insights
        </div>

        <div className="sidebar-bottom">
          <div className="sidebar-status">
            <span className={`status-dot ${apiOnline ? "online" : ""}`} />
            <span>{apiOnline ? "API connected" : "API disconnected"}</span>
          </div>
          <p>ChurnIQ · Development version</p>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div>
            <p className="eyebrow">CUSTOMER ANALYTICS</p>
            <h1>Churn Prediction Dashboard</h1>
            <p className="subtitle">
              Understand customer risk and identify retention opportunities.
            </p>
          </div>

          <div className={`connection-pill ${apiOnline ? "connected" : ""}`}>
            {apiOnline ? <Wifi size={17} /> : <WifiOff size={17} />}
            {apiOnline ? "Backend connected" : "Backend offline"}
          </div>
        </header>

        <section className="metrics-grid">
          <div className="metric-card">
            <div className="metric-heading">
              <span>Predictions this session</span>
              <div className="metric-icon blue">
                <Users size={20} />
              </div>
            </div>
            <div className="metric-value">{history.length}</div>
            <p className="metric-note">Current browser session</p>
          </div>

          <div className="metric-card">
            <div className="metric-heading">
              <span>Latest churn risk</span>
              <div className="metric-icon purple">
                <TrendingUp size={20} />
              </div>
            </div>
            <div className="metric-value">
              {prediction ? `${prediction.churn_risk_percent}%` : "—"}
            </div>
            <p className="metric-note">Most recent prediction</p>
          </div>

          <div className="metric-card">
            <div className="metric-heading">
              <span>High-risk predictions</span>
              <div className="metric-icon orange">
                <AlertTriangle size={20} />
              </div>
            </div>
            <div className="metric-value">{highRiskCount}</div>
            <p className="metric-note">In this session only</p>
          </div>

          <div className="metric-card">
            <div className="metric-heading">
              <span>API status</span>
              <div className="metric-icon green">
                <ShieldCheck size={20} />
              </div>
            </div>
            <div className="metric-value status-value">
              {apiOnline ? "Online" : "Offline"}
            </div>
            <p className="metric-note">FastAPI · Port 8000</p>
          </div>
        </section>

        <div className="dashboard-grid">
          <section
            id="customer-analysis"
            className="panel form-panel"
            style={{ scrollMarginTop: "24px" }}
          >
            <div className="panel-heading">
              <div>
                <h2>Analyze a customer</h2>
                <p>Enter customer details to estimate churn risk.</p>
              </div>
              <div className="heading-icon">
                <Users size={20} />
              </div>
            </div>

            <form onSubmit={predictChurn}>
              <label htmlFor="tenure">
                Customer tenure
                <span className="label-hint">Months</span>
              </label>
              <input
                id="tenure"
                type="number"
                min="0"
                max="120"
                value={customer.tenure}
                onChange={(e) => setCustomer(prev => ({ ...prev, tenure: Number(e.target.value) }))}
                required
              />

              <label htmlFor="monthly">
                Monthly charges
                <span className="label-hint">Amount per month</span>
              </label>
              <input
                id="monthly"
                type="number"
                min="0"
                step="0.01"
                value={customer.monthly_charges}
                onChange={(e) => setCustomer(prev => ({ ...prev, monthly_charges: Number(e.target.value) }))}
                required
              />

              <label htmlFor="contract">Contract type</label>
              <select
                id="contract"
                value={customer.contract}
                onChange={(e) => setCustomer(prev => ({ ...prev, contract: e.target.value }))}
              >
                <option value="Monthly">Monthly</option>
                <option value="Annual">Annual</option>
              </select>

              <button
                className="primary-button"
                type="submit"
                disabled={loading || !apiOnline}
              >
                {loading ? (
                  <>
                    <RefreshCw className="spin" size={18} />
                    Analyzing customer...
                  </>
                ) : (
                  <>
                    <Activity size={18} />
                    Predict churn risk
                  </>
                )}
              </button>

              {!apiOnline && (
                <p className="form-hint">
                  Start the FastAPI backend before predicting.
                </p>
              )}
            </form>

            {error && (
              <div className="error-message" role="alert">
                <AlertTriangle size={18} />
                <span>{error}</span>
              </div>
            )}
          </section>

          <section className="panel result-panel">
            <div className="panel-heading">
              <div>
                <h2>Prediction results</h2>
                <p>Risk assessment for the submitted customer.</p>
              </div>
              {prediction && (
                <span
                  className={`risk-badge ${riskClass(prediction.risk_segment)}`}
                >
                  {prediction.risk_segment} risk
                </span>
              )}
            </div>

            {!prediction ? (
              <div className="empty-state">
                <div className="empty-icon">
                  <Activity size={30} />
                </div>
                <h3>Ready to analyze</h3>
                <p>
                  Enter customer details and click Predict churn risk to view the
                  result here.
                </p>
              </div>
            ) : (
              <>
                <div className="risk-score">
                  <div>
                    <span className="score-label">Estimated churn risk</span>
                    <div className="score-number">
                      {prediction.churn_risk_percent}
                      <span>%</span>
                    </div>
                  </div>

                  <div
                    className={`score-symbol ${riskClass(
                      prediction.risk_segment
                    )}`}
                  >
                    {prediction.risk_segment === "Low" ? (
                      <CheckCircle2 size={30} />
                    ) : (
                      <AlertTriangle size={30} />
                    )}
                  </div>
                </div>

                <div className="progress-track">
                  <div
                    className={`progress-fill ${riskClass(
                      prediction.risk_segment
                    )}`}
                    style={{
                      width: `${Math.min(
                        100,
                        Math.max(0, prediction.churn_risk_percent)
                      )}%`,
                    }}
                  />
                </div>

                <div className="progress-labels">
                  <span>0% · Low</span>
                  <span>100% · High</span>
                </div>

                <div className="recommendation-box">
                  {prediction.shap_explanation && prediction.shap_explanation.length > 0 && (
                    <section className="shap-panel">
                      <div className="shap-heading">
                        <div>
                          <h3>Why is this customer at risk?</h3>
                          <p>
                            SHAP explains how each feature moves the model's
                            output from its baseline.
                          </p>
                        </div>
                      </div>

                      <ShapWaterfall
                        baseValue={prediction.shap_base_value}
                        outputValue={prediction.shap_output_value}
                        contributions={prediction.shap_explanation}
                      />
                    </section>
                  )}
                  <div className="recommendation-title">
                    <Lightbulb size={19} />
                    Retention recommendation
                  </div>
                  <p>{prediction.recommendation}</p>
                </div>

                <div className="model-note">
                  <strong>Prediction method:</strong> {prediction.model_type}
                  <p>Feature explanations generated using SHAP and LIME.</p>
                </div>

                <button
                  type="button"
                  className="primary-button"
                  onClick={downloadPredictionPdf}
                  disabled={!prediction}
                >
                  Download PDF Report
                </button>
              </>
            )}
          </section>
        </div>

        <section className="panel history-panel batch-panel">
          <div className="panel-heading">
            <div>
              <h2>Batch customer analysis</h2>
              <p>Upload a CSV to predict churn risk for multiple customers.</p>
            </div>
            <div className="heading-icon"><Users size={20} /></div>
          </div>

          <div className="batch-upload-controls">
            <label htmlFor="customer-csv">Customer CSV file</label>
            <input
              id="customer-csv"
              type="file"
              accept=".csv,text/csv"
              onChange={(event) => {
                setSelectedCsv(event.target.files?.[0] || null);
                setBatchError("");
              }}
            />
            <p className="form-hint">
              Expected columns: tenure, monthly_charges, contract, internet_fiber,
              tech_support, paperless_billing, senior_citizen, dependents.
            </p>
            <button
              type="button"
              className="primary-button"
              onClick={uploadCsv}
              disabled={batchLoading || !apiOnline || !selectedCsv}
            >
              {batchLoading ? <><RefreshCw className="spin" size={18} /> Processing CSV...</> : "Upload CSV and predict"}
            </button>
            {batchError && <div className="error-message" role="alert"><AlertTriangle size={18} /><span>{batchError}</span></div>}
          </div>

          {batchResult && (
            <>
              <div className="metrics-grid batch-metrics">
                <div className="metric-card"><div className="metric-heading"><span>Total customers</span><Users size={18} /></div><div className="metric-value">{batchResult.total_customers}</div></div>
                <div className="metric-card"><div className="metric-heading"><span>High risk</span><AlertTriangle size={18} /></div><div className="metric-value">{batchResult.high_risk}</div></div>
                <div className="metric-card"><div className="metric-heading"><span>Medium risk</span><Activity size={18} /></div><div className="metric-value">{batchResult.medium_risk}</div></div>
                <div className="metric-card"><div className="metric-heading"><span>Low risk</span><CheckCircle2 size={18} /></div><div className="metric-value">{batchResult.low_risk}</div></div>
              </div>
              <div className="table-scroll">
                <table>
                  <thead><tr><th>CSV row</th><th>Churn risk</th><th>Risk segment</th></tr></thead>
                  <tbody>
                    {(batchResult.customers || []).map((item, index) => (
                      <tr key={`${item.customer_index}-${index}`}>
                        <td>{item.customer_index}</td>
                        <td>{Number(item.churn_risk_percent).toFixed(2)}%</td>
                        <td><span className={`risk-badge ${riskClass(item.risk_segment)}`}>{item.risk_segment}</span></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <button type="button" className="primary-button" onClick={downloadBatchPdf}>Download batch PDF report</button>
            </>
          )}
        </section>

        <section className="panel history-panel evaluation-panel">
          <div className="panel-heading">
            <div><h2>Model evaluation</h2><p>Review the evaluation metrics reported by your backend.</p></div>
            <div className="heading-icon"><TrendingUp size={20} /></div>
          </div>
          <button type="button" className="primary-button" onClick={loadEvaluation} disabled={metricsLoading || !apiOnline}>
            {metricsLoading ? <><RefreshCw className="spin" size={18} /> Loading metrics...</> : "Load model evaluation"}
          </button>
          {metricsError && <div className="error-message" role="alert"><AlertTriangle size={18} /><span>{metricsError}</span></div>}
          {rocMetrics && (
            <div className="table-scroll">
              <table>
                <tbody>
                  <tr><th>Evaluation type</th><td>{rocMetrics.evaluation_type || "Not specified"}</td></tr>
                  <tr><th>Test customers</th><td>{rocMetrics.test_customers ?? "—"}</td></tr>
                  <tr><th>ROC-AUC</th><td>{typeof rocMetrics.roc_auc === "number" ? rocMetrics.roc_auc.toFixed(4) : "—"}</td></tr>
                  <tr><th>Accuracy</th><td>{typeof rocMetrics.accuracy === "number" ? `${(rocMetrics.accuracy <= 1 ? rocMetrics.accuracy * 100 : rocMetrics.accuracy).toFixed(2)}%` : "—"}</td></tr>
                  <tr><th>Churn rate</th><td>{typeof rocMetrics.churn_rate_percent === "number" ? `${rocMetrics.churn_rate_percent.toFixed(2)}%` : "—"}</td></tr>
                </tbody>
              </table>
              {rocMetrics.note && <p className="form-hint">{rocMetrics.note}</p>}
              <button
                type="button"
                className="primary-button"
                onClick={downloadEvaluationPdf}
                style={{ marginTop: "12px" }}
              >
                Download evaluation PDF with charts
              </button>
            </div>
          )}
        </section>

        <section
          id="retention-insights"
          className="panel history-panel"
          style={{ marginBottom: "24px", scrollMarginTop: "24px" }}
        >
          <div className="panel-heading">
            <div>
              <h2>Smart Retention Insights</h2>
              <p>Generate practical action suggestions for the latest predicted customer.</p>
            </div>
            <div className="heading-icon">
              <Lightbulb size={20} />
            </div>
          </div>

          {!prediction && (
            <div className="history-empty">
              Analyze a customer first, then generate retention recommendations.
            </div>
          )}

          <button
            type="button"
            className="primary-button"
            onClick={loadRetentionRecommendations}
            disabled={!prediction || retentionLoading || !apiOnline}
          >
            {retentionLoading ? (
              <><RefreshCw className="spin" size={18} /> Generating retention plan...</>
            ) : (
              <><Lightbulb size={18} /> Generate Retention Plan</>
            )}
          </button>

          {retentionError && (
            <div className="error-message" role="alert" style={{ marginTop: "14px" }}>
              <AlertTriangle size={18} />
              <span>{retentionError}</span>
            </div>
          )}

          {retentionResult && (
            <div style={{ marginTop: "24px" }}>
              <div className="metrics-grid">
                <div className="metric-card">
                  <div className="metric-heading"><span>Risk segment</span></div>
                  <div className="metric-value">{retentionResult.risk_segment ?? "—"}</div>
                </div>
                <div className="metric-card">
                  <div className="metric-heading"><span>Churn probability</span></div>
                  <div className="metric-value">
                    {typeof retentionResult.churn_probability_percent === "number"
                      ? `${retentionResult.churn_probability_percent.toFixed(2)}%`
                      : "—"}
                  </div>
                </div>
                <div className="metric-card">
                  <div className="metric-heading"><span>Action priority</span></div>
                  <div className="metric-value" style={{ fontSize: "18px" }}>
                    {retentionResult.priority ?? "—"}
                  </div>
                </div>
              </div>

              <h3 style={{ marginTop: "24px" }}>Recommended actions</h3>
              <div style={{ display: "grid", gap: "14px" }}>
                {(retentionResult.recommendations ?? []).map(
                  (item, index) => (
                    <div className="recommendation-box" key={`${item.action}-${index}`}>
                      <div className="recommendation-title">
                        <CheckCircle2 size={19} />
                        {item.action}
                      </div>
                      <p>{item.reason}</p>
                      <span className={`risk-badge ${riskClass(item.priority)}`}>
                        {item.priority} priority
                      </span>
                    </div>
                  )
                )}
              </div>

              {retentionResult.disclaimer && (
                <p className="form-hint" style={{ marginTop: "16px" }}>
                  {retentionResult.disclaimer}
                </p>
              )}
            </div>
          )}
        </section>

        <section className="panel history-panel">
          <div className="panel-heading">
            <div>
              <h2>Recent prediction activity</h2>
              <p>Predictions generated during this session.</p>
            </div>
          </div>

          {history.length === 0 ? (
            <div className="history-empty">
              No predictions yet. Analyze a customer to populate this section.
            </div>
          ) : (
            <>
              <div className="chart-wrap">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={chartData}>
                    <CartesianGrid
                      strokeDasharray="3 3"
                      vertical={false}
                      stroke="#e8edf4"
                    />
                    <XAxis dataKey="name" tickLine={false} axisLine={false} />
                    <YAxis
                      domain={[0, 100]}
                      tickLine={false}
                      axisLine={false}
                      tickFormatter={(value) => `${value}%`}
                    />
                    <Tooltip
                      formatter={(value) => [`${value}%`, "Churn risk"]}
                    />
                    <Bar
                      dataKey="risk"
                      fill="#4f46e5"
                      radius={[6, 6, 0, 0]}
                      maxBarSize={55}
                    />
                  </BarChart>
                </ResponsiveContainer>
              </div>

              <div className="table-scroll">
                <table>
                  <thead>
                    <tr>
                      <th>Time</th>
                      <th>Tenure</th>
                      <th>Monthly charges</th>
                      <th>Contract</th>
                      <th>Risk score</th>
                      <th>Risk segment</th>
                    </tr>
                  </thead>
                  <tbody>
                    {history.map((item) => (
                      <tr key={item.id}>
                        <td>{item.date}</td>
                        <td>{item.customer.tenure} months</td>
                        <td>{item.customer.monthly_charges.toFixed(2)}</td>
                        <td>{item.customer.contract}</td>
                        <td>{item.risk}%</td>
                        <td>
                          <span
                            className={`risk-badge ${riskClass(item.segment)}`}
                          >
                            {item.segment}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}
        </section>

        <footer className="footer">
          <span>ChurnIQ · Customer analytics prototype</span>
          <span>
            <span className={`status-dot ${apiOnline ? "online" : ""}`} />
            {apiOnline ? "API connected" : "Waiting for backend"}
          </span>
        </footer>
      </main>
    </div>
  );
}

export default App;